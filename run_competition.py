#!/usr/bin/env python3
"""Seven independent policies acting together in one shared, competitive world."""
import argparse, csv, hashlib, json, os, shutil, signal, subprocess, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from sdaea_streaming_v2 import Config, Learner, SensoryMemory, body_signal, motor_table
from sdaea_online_validate import ActionSpec, extract_hp, seed_everything

VARIANTS = [(48,96,2),(48,256,2),(96,96,2),(96,256,2),(144,256,2),(96,256,3)]
LABELS = [f'{r}_w{w}_d{d}' for r,w,d in VARIANTS]+['random']
ROOT=Path(__file__).resolve().parent

def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def reset_indices(env, indices):
    env._send_as_json(dict(type='reset',agent_indices=indices))
    reply=env._get_json_dict()
    assert reply['type']=='reset'
    return env._process_obs(reply['obs'])

def ensure_live(env, obs):
    for _ in range(10):
        dead=[i for i,o in enumerate(obs) if extract_hp(o,'hp')<=0]
        if not dead: return obs
        before=[extract_hp(o,'hp') for o in obs]
        obs=reset_indices(env,dead)
        for i in range(len(obs)):
            if i not in dead and extract_hp(obs[i],'hp')!=before[i]:
                raise RuntimeError('Selective reset changed live HP')
    raise RuntimeError('Repeated dead respawn observations')

def worker(a):
    from godot_rl.core.godot_env import GodotEnv
    torch.set_num_threads(2)
    env=GodotEnv(env_path=None,port=a.port,show_window=True,seed=19)
    learners=[]; memories=[]; configs=[]
    specs=[ActionSpec(name=k,size=int(v.n),learned='shoot' not in k.lower(),fixed_value=0)
           for k,v in env.action_spaces[0].spaces.items()]
    assert all(space == env.action_spaces[0] for space in env.action_spaces)
    table=motor_table(specs)
    assert env.num_envs==7 and list(env.agent_policy_names)==LABELS, env.agent_policy_names
    for i,(res,width,depth) in enumerate(VARIANTS+[ (96,256,2) ]):
        seed_everything(19)
        c=Config(image_size=res,width=width,depth=depth,device='cuda',seed=19,
                 no_learn=True,random_actions=i==6)
        l=Learner(c,len(table),torch.device('cuda'))
        learners.append(l);configs.append(c);memories.append(SensoryMemory(c,torch.device('cuda'),len(table)))
        (a.run_dir/LABELS[i]).mkdir()
        l.save(a.run_dir/LABELS[i]/'initial.pt',specs,0)
    # Independent random generators prevent one policy's sampling from consuming another's draws.
    rngs=[np.random.default_rng(1000+i) for i in range(7)]
    obs,_=env.reset()
    obs=ensure_live(env,obs)
    states=[m.state(o,None,0) for m,o in zip(memories,obs)]
    plan=dict(labels=LABELS,configs=[asdict(c) for c in configs],
              phases=[['initial',a.eval_steps],['train',a.train_steps],['trained',a.eval_steps]],
              shared_scene=True,notes='One continuous shared arena. Phase comparisons also reflect evolving opponents/world. Random actions uniform over same 27 actions, shooting disabled for all. Exact block counters are diagnostics only, never policy inputs or learning targets.')
    write(a.run_dir/'plan.json',plan)
    total=0;results={};stop=False;reset_checks=0
    def interrupt(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    fields=['step','phase','agent','elapsed','hp','terminal','valence','positive_hp_events','negative_hp_events','updates','action','green_blocks','red_blocks']
    try:
      with (a.run_dir/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for phase,length in plan['phases']:
          for i,l in enumerate(learners):
            l.c.no_learn=phase!='train' or i==6;l.clear()
          frozen=[{k:v.detach().clone() for k,v in l.actor.state_dict().items()} for l in learners]
          frozen_critics=[{k:v.detach().clone() for k,v in l.critic.state_dict().items()} for l in learners]
          stats=[dict(steps=0,deaths=0,positive_hp_events=0,negative_hp_events=0,hp_sum=0.,updates=0) for _ in LABELS]
          start_counts=[np.asarray(o['block_counts']).copy() for o in obs]
          elapsed_phase=0
          while elapsed_phase<length and not stop:
            actions=[]
            for i,l in enumerate(learners):
              if i==6: actions.append(int(rngs[i].integers(len(table))))
              else:
                with torch.no_grad(): probs=l.distribution(states[i]).probs[0].cpu().numpy().astype(float)
                actions.append(int(rngs[i].choice(len(table),p=probs/probs.sum())))
            values=[0.]*7;pos=[0]*7;neg=[0]*7;term=[False]*7
            previous=[extract_hp(o,'hp') for o in obs]
            for j in range(min(4,length-elapsed_phase)):
              obs,_,done,truncated,_=env.step([table[x] for x in actions],order_ij=True)
              if len(obs)!=7: raise RuntimeError('Agent count changed')
              event=False
              for i,o in enumerate(obs):
                hp=extract_hp(o,'hp');term[i]=bool(done[i] or truncated[i] or hp<=0)
                if term[i] and hp>0: raise RuntimeError('Unexpected live terminal')
                values[i]+=configs[i].gamma**j*body_signal(previous[i],hp,term[i],configs[i])['valence']
                change=hp-previous[i];pos[i]+=int(change>.1);neg[i]+=int(change<-.1)
                event |= abs(change)>=1.25
                stats[i]['hp_sum']+=max(hp,0);previous[i]=hp
              total+=1;elapsed_phase+=1
              if any(term) or event: break
            elapsed=j+1
            for i,l in enumerate(learners):
              nxt=None if term[i] else memories[i].state(obs[i],actions[i],elapsed)
              l.learn(states[i],actions[i],values[i],nxt,elapsed,term[i])
              t=stats[i];t['steps']+=elapsed;t['deaths']+=int(term[i]);t['positive_hp_events']+=pos[i];t['negative_hp_events']+=neg[i];t['updates']+=int(not l.c.no_learn)
              w.writerow(dict(step=total,phase=phase,agent=LABELS[i],elapsed=elapsed,hp=previous[i],terminal=int(term[i]),valence=values[i],positive_hp_events=pos[i],negative_hp_events=neg[i],updates=t['updates'],action=actions[i],green_blocks=int(obs[i]['block_counts'][0]),red_blocks=int(obs[i]['block_counts'][1])))
              states[i]=nxt
            if any(term):
              reset_obs=ensure_live(env,reset_indices(env,[i for i,t in enumerate(term) if t]))
              reset_checks+=1
              for i in range(7):
                if term[i]:
                  if extract_hp(reset_obs[i],'hp')<=0: raise RuntimeError('Dead body failed to respawn')
                  learners[i].clear();memories[i].clear();states[i]=memories[i].state(reset_obs[i],None,0)
                elif extract_hp(reset_obs[i],'hp')!=previous[i]:
                  raise RuntimeError('Reset changed a live competitor HP')
              for i in range(7):
                if not term[i]:
                  states[i]=memories[i].state(reset_obs[i],actions[i],0)
              obs=reset_obs
            if total//500 != (total-elapsed)//500:
              f.flush();write(a.run_dir/'status.json',dict(status='running',phase=phase,step=total,phase_step=elapsed_phase,total_steps=a.train_steps+2*a.eval_steps,updated_at=time.time()))
              print(f'phase={phase} shared_step={total}',flush=True)
            if total//10000 != (total-elapsed)//10000:
              for i,l in enumerate(learners): l.save(a.run_dir/LABELS[i]/'latest.pt',specs,total)
          if stop: raise KeyboardInterrupt('Stopped')
          for i,l in enumerate(learners):
            if phase!='train' or i==6:
              assert all(torch.equal(v,l.actor.state_dict()[k]) for k,v in frozen[i].items())
              assert all(torch.equal(v,l.critic.state_dict()[k]) for k,v in frozen_critics[i].items())
            l.save(a.run_dir/LABELS[i]/f'{phase}.pt',specs,total)
            l.save(a.run_dir/LABELS[i]/'latest.pt',specs,total)
            counts=np.asarray(obs[i]['block_counts'])-start_counts[i]
            stats[i]['green_blocks']=int(counts[0]);stats[i]['red_blocks']=int(counts[1])
            stats[i]['mean_hp']=stats[i].pop('hp_sum')/stats[i]['steps']
            stats[i]['deaths_per_10000_steps']=10000*stats[i]['deaths']/stats[i]['steps']
          results[phase]=dict(zip(LABELS,stats));write(a.run_dir/'comparison.json',results)
        write(a.run_dir/'status.json',dict(status='complete',step=total,finished_at=time.time(),selective_reset_checks=reset_checks))
    except BaseException as e:
        for i,l in enumerate(learners): l.save(a.run_dir/LABELS[i]/'interrupted.pt',specs,total)
        write(a.run_dir/'status.json',dict(status='failed',step=total,error=str(e)));raise
    finally: env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--port',type=int,default=11088);p.add_argument('--train-steps',type=int,default=100000);p.add_argument('--eval-steps',type=int,default=20000);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker: return worker(a)
    a.run_dir.mkdir(parents=True,exist_ok=False)
    write(a.run_dir/'status.json',dict(status='starting',started_at=time.time()))
    sources=[ROOT/'run_competition.py',ROOT/'sdaea_streaming_v2.py',ROOT/'sdaea_online_validate.py']
    sources += [p for p in (ROOT.parent/'EnvolutionRobot').rglob('*')
                if p.is_file() and p.suffix in ('.gd','.tscn','.godot')
                and not any(part.startswith('.') for part in p.relative_to(ROOT.parent).parts)]
    hashes={}
    for source in sources:
        relative=source.relative_to(ROOT.parent)
        target=a.run_dir/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    write(a.run_dir/'source_hashes.json',hashes)
    game=None;trainer=None
    env=os.environ.copy();env['DOTNET_ROOT']='/home/gao/Documents/Codex/2026-09-23-godot-net/.tools/dotnet';env['PATH']=env['DOTNET_ROOT']+':'+env['PATH']
    godot='/home/gao/Documents/Codex/2026-09-23-godot-net/.tools/godot/Godot_v4.7.2-stable_mono_linux_x86_64/Godot_v4.7.2-stable_mono_linux.x86_64'
    try:
      with (a.run_dir/'python.log').open('w') as out,(a.run_dir/'godot.log').open('w') as gout:
        trainer=subprocess.Popen([sys.executable,'-u',__file__,*sys.argv[1:],'--worker'],stdout=out,stderr=subprocess.STDOUT,env=env)
        deadline=time.monotonic()+90
        while 'waiting for remote GODOT connection' not in (a.run_dir/'python.log').read_text():
          if trainer.poll() is not None or time.monotonic()>deadline: raise RuntimeError('Trainer listener failed')
          time.sleep(.25)
        game=subprocess.Popen([godot,'--path',str(ROOT.parent/'EnvolutionRobot'),'res://scenes/training_scene/competition.tscn',f'--port={a.port}','--env_seed=19'],stdout=gout,stderr=subprocess.STDOUT,env=env)
        last=time.monotonic();size=-1
        while trainer.poll() is None:
          time.sleep(2)
          f=a.run_dir/'metrics.csv';new=f.stat().st_size if f.exists() else 0
          if new!=size: size=new;last=time.monotonic()
          if time.monotonic()-last>600: raise RuntimeError('No progress for 10 minutes')
          if game.poll() is not None:
            trainer.wait(timeout=15)
        if trainer.returncode: raise RuntimeError(f'Trainer exited {trainer.returncode}')
    except BaseException as e:
      previous=json.loads((a.run_dir/'status.json').read_text())
      write(a.run_dir/'status.json',{**previous,'status':'failed','launcher_error':str(e)});raise
    finally:
      for proc in (trainer,game):
        if proc is not None and proc.poll() is None:
          proc.terminate()
          try: proc.wait(timeout=30)
          except subprocess.TimeoutExpired: proc.kill();proc.wait()
if __name__=='__main__': main()
