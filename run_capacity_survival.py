#!/usr/bin/env python3
"""Capacity expansion and continued learning in one shared competitive world."""
import argparse, csv, hashlib, json, os, shutil, signal, subprocess, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from sdaea_streaming_v2 import Config, Learner, SensoryMemory, body_signal, motor_table
from sdaea_online_validate import ActionSpec, extract_hp, seed_everything

VARIANTS = [(96,256,3),(96,512,3),(96,768,3),(96,256,4),(96,256,5),(96,512,5)]
LABELS = [f'{r}_w{w}_d{d}' for r,w,d in VARIANTS]+['96_w256_d3_frozen','random']
N = len(LABELS)
FROZEN = {N-2,N-1}
ROOT=Path(__file__).resolve().parent

def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def reset_indices(env, indices):
    frame=getattr(env,'physics_frame',None)
    env._send_as_json(dict(type='reset',agent_indices=indices))
    reply=env._get_json_dict()
    assert reply['type']=='reset'
    if frame is not None: assert env.physics_frame==frame, 'Selective reset advanced physics'
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
    if a.fast:
        from fast_godot_env import FastGodotEnv
        GodotEnv=FastGodotEnv
    env=GodotEnv(env_path=None,port=a.port,show_window=True,seed=19)
    from grow_policy import grow_network
    source=torch.load(a.source_checkpoint,map_location='cpu',weights_only=True)
    assert source['config']['width']==256 and source['config'].get('depth',2)==3
    learners=[]; memories=[]; configs=[]
    specs=[ActionSpec(name=k,size=int(v.n),learned='shoot' not in k.lower(),fixed_value=0)
           for k,v in env.action_spaces[0].spaces.items()]
    assert all(space == env.action_spaces[0] for space in env.action_spaces)
    table=motor_table(specs)
    assert source['specs']==[asdict(s) for s in specs]
    assert env.num_envs==N and list(env.agent_policy_names)==LABELS, env.agent_policy_names
    for i,(res,width,depth) in enumerate(VARIANTS+[(96,256,3),(96,256,3)]):
        seed_everything(19)
        c=Config(image_size=res,width=width,depth=depth,eye_height=a.eye_height,eye_width=a.eye_width,device='cuda',seed=19,
                 no_learn=True,random_actions=i==N-1)
        l=Learner(c,len(table),torch.device('cuda'))
        learners.append(l);configs.append(c);memories.append(SensoryMemory(c,torch.device('cuda'),len(table)))
        if i in (0,N-2):
            l.load(a.source_checkpoint,specs)
        elif i!=N-1:
            for model in ('actor','critic'):
                grow_network(getattr(l,model),source[model],256,3)
            l.clear()
        (a.run_dir/LABELS[i]).mkdir()
        l.save(a.run_dir/LABELS[i]/'initial.pt',specs,0)
    # Independent random generators prevent one policy's sampling from consuming another's draws.
    rngs=[np.random.default_rng(1000+i) for i in range(N)]
    obs,_=env.reset()
    obs=ensure_live(env,obs)
    if a.fast and a.verify_visuals:
        legacy=env.call('get_obs')
        for current,old in zip(obs,legacy):
            for eye in ('left_eye','right_eye'):
                assert current[eye].tobytes()==bytes.fromhex(old[eye]), 'Raw and hex pixels differ'
    states=[m.state(o,None,0) for m,o in zip(memories,obs)]
    plan=dict(labels=LABELS,configs=[asdict(c) for c in configs],
              phases=[['initial',a.initial_eval_steps],['train',a.train_steps],['trained',a.eval_steps]],
              source_checkpoint=str(a.source_checkpoint),shared_scene=True,fast=a.fast,
              parameter_counts={label:sum(p.numel() for model in (l.actor,l.critic) for p in model.parameters()) for label,l in zip(LABELS,learners)},
              notes='Six learning models, one frozen source policy, one uniform random policy in one arena. Base resumes exact weights. Wider models replicate pretrained neurons with tiny symmetry-breaking noise; extra normalized layers start as identity linear maps but change the initial policy. Each phase restarts all bodies at HP6 without a physics step, clearing memory/traces. Initial evaluation measures transfer effects; common resources/opponents evolve. Single seed, not independent replicates. No environment reward or block labels used for learning.')
    write(a.run_dir/'plan.json',plan)
    total=0;results={};timings={};stop=False;reset_checks=0;visual_changes=[0]*N
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
            l.c.no_learn=phase!='train' or i in FROZEN;l.clear()
          # Equal body energy at the start of every phase, including frozen evaluation.
          frame=getattr(env,'physics_frame',None)
          assert all(env.call('restart_competition_body'))
          obs=env.observe()
          assert frame==env.physics_frame
          assert all(abs(extract_hp(o,'hp')-6.)<1e-6 for o in obs)
          for i in range(N):
            memories[i].clear();states[i]=memories[i].state(obs[i],None,0)
          lives=[0]*N;lifetimes=[[] for _ in range(N)];first_death=[None]*N
          frozen=[{k:v.detach().clone() for k,v in l.actor.state_dict().items()} for l in learners]
          frozen_critics=[{k:v.detach().clone() for k,v in l.critic.state_dict().items()} for l in learners]
          stats=[dict(steps=0,deaths=0,positive_hp_events=0,negative_hp_events=0,hp_sum=0.,updates=0,no_acceleration_steps=0) for _ in LABELS]
          start_counts=[np.asarray(o['block_counts']).copy() for o in obs]
          phase_started=time.perf_counter();phase_bytes=getattr(env,'bytes_received',0);phase_frames=getattr(env,'rgb_frames',0)
          elapsed_phase=0
          while elapsed_phase<length and not stop:
            actions=[]
            for i,l in enumerate(learners):
              if i==N-1: actions.append(int(rngs[i].integers(len(table))))
              else:
                with torch.no_grad(): probs=l.distribution(states[i]).probs[0].cpu().numpy().astype(float)
                actions.append(int(rngs[i].choice(len(table),p=probs/probs.sum())))
            values=[0.]*N;pos=[0]*N;neg=[0]*N;term=[False]*N
            previous=[extract_hp(o,'hp') for o in obs]
            hold=min(4,length-elapsed_phase)
            for j in range(hold):
              extra={'include_rgb': j==hold-1} if a.fast else {}
              obs,_,done,truncated,_=env.step([table[x] for x in actions],order_ij=True,**extra)
              if len(obs)!=N: raise RuntimeError('Agent count changed')
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
            if a.fast and 'left_eye' not in obs[0]:
              snapshot=env.observe()
              assert all(extract_hp(o,'hp')==previous[i] for i,o in enumerate(snapshot)), 'Snapshot advanced physics'
              obs=snapshot
            for i,l in enumerate(learners):
              nxt=None if term[i] else memories[i].state(obs[i],actions[i],elapsed)
              if a.verify_visuals and nxt is not None:
                visual_changes[i]+=int(not torch.equal(states[i].image,nxt.image))
              l.learn(states[i],actions[i],values[i],nxt,elapsed,term[i])
              lives[i]+=elapsed
              if term[i]:
                lifetimes[i].append(lives[i]);lives[i]=0
                if first_death[i] is None: first_death[i]=elapsed_phase
              action_dict=dict(zip([s.name for s in specs],table[actions[i]]))
              stats[i]['no_acceleration_steps']+=elapsed*int(action_dict['accelerate_forward']==1 and action_dict['accelerate_sideways']==1)
              t=stats[i];t['steps']+=elapsed;t['deaths']+=int(term[i]);t['positive_hp_events']+=pos[i];t['negative_hp_events']+=neg[i];t['updates']+=int(not l.c.no_learn)
              w.writerow(dict(step=total,phase=phase,agent=LABELS[i],elapsed=elapsed,hp=previous[i],terminal=int(term[i]),valence=values[i],positive_hp_events=pos[i],negative_hp_events=neg[i],updates=t['updates'],action=actions[i],green_blocks=int(obs[i]['block_counts'][0]),red_blocks=int(obs[i]['block_counts'][1])))
              states[i]=nxt
            if any(term):
              reset_obs=ensure_live(env,reset_indices(env,[i for i,t in enumerate(term) if t]))
              reset_checks+=1
              for i in range(N):
                if term[i]:
                  if extract_hp(reset_obs[i],'hp')<=0: raise RuntimeError('Dead body failed to respawn')
                  learners[i].clear();memories[i].clear();states[i]=memories[i].state(reset_obs[i],None,0)
                elif extract_hp(reset_obs[i],'hp')!=previous[i]:
                  raise RuntimeError('Reset changed a live competitor HP')
              for i in range(N):
                if not term[i]:
                  states[i]=memories[i].state(reset_obs[i],actions[i],0)
              obs=reset_obs
            if total//500 != (total-elapsed)//500:
              f.flush();write(a.run_dir/'status.json',dict(status='running',phase=phase,step=total,phase_step=elapsed_phase,total_steps=a.initial_eval_steps+a.train_steps+a.eval_steps,updated_at=time.time()))
              print(f'phase={phase} shared_step={total}',flush=True)
            if total//10000 != (total-elapsed)//10000:
              for i,l in enumerate(learners): l.save(a.run_dir/LABELS[i]/'latest.pt',specs,total)
          if stop: raise KeyboardInterrupt('Stopped')
          for i,l in enumerate(learners):
            if phase!='train' or i in FROZEN:
              assert all(torch.equal(v,l.actor.state_dict()[k]) for k,v in frozen[i].items())
              assert all(torch.equal(v,l.critic.state_dict()[k]) for k,v in frozen_critics[i].items())
            l.save(a.run_dir/LABELS[i]/f'{phase}.pt',specs,total)
            l.save(a.run_dir/LABELS[i]/'latest.pt',specs,total)
            counts=np.asarray(obs[i]['block_counts'])-start_counts[i]
            stats[i]['green_blocks']=int(counts[0]);stats[i]['red_blocks']=int(counts[1])
            stats[i].update(initial_hp=6.,final_hp=extract_hp(obs[i],'hp'),completed_lifetimes=lifetimes[i],censored_lifetime=lives[i],longest_lifetime=max(lifetimes[i]+[lives[i]]),first_death_step=first_death[i],survived_entire_phase=first_death[i] is None)
            stats[i]['no_acceleration_fraction']=stats[i].pop('no_acceleration_steps')/stats[i]['steps']
            stats[i]['mean_hp']=stats[i].pop('hp_sum')/stats[i]['steps']
            stats[i]['deaths_per_10000_steps']=10000*stats[i]['deaths']/stats[i]['steps']
          results[phase]=dict(zip(LABELS,stats));write(a.run_dir/'comparison.json',results)
          timings[phase]=dict(steps=length,seconds=time.perf_counter()-phase_started,bytes_received=getattr(env,'bytes_received',0)-phase_bytes,rgb_observations=getattr(env,'rgb_frames',0)-phase_frames)
          write(a.run_dir/'timings.json',timings)
        if a.verify_visuals:
            assert all(n>0 for n in visual_changes), 'A camera never changed'
            write(a.run_dir/'visual_checks.json',dict(zip(LABELS,visual_changes)))
        write(a.run_dir/'status.json',dict(status='complete',step=total,finished_at=time.time(),selective_reset_checks=reset_checks))
    except BaseException as e:
        for i,l in enumerate(learners): l.save(a.run_dir/LABELS[i]/'interrupted.pt',specs,total)
        write(a.run_dir/'status.json',dict(status='failed',step=total,error=str(e)));raise
    finally: env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--port',type=int,default=11088);p.add_argument('--train-steps',type=int,default=50000);p.add_argument('--eval-steps',type=int,default=50000);p.add_argument('--worker',action='store_true');p.add_argument('--fast',action='store_true');p.add_argument('--uncapped',action='store_true');p.add_argument('--source-checkpoint',type=Path,required=True);p.add_argument('--initial-eval-steps',type=int,default=10000);p.add_argument('--verify-visuals',action='store_true');p.add_argument('--eye-width',type=int,default=160);p.add_argument('--eye-height',type=int,default=150);a=p.parse_args()
    if min(a.eye_width,a.eye_height)<36: p.error('Eye dimensions must be >=36')
    if not a.fast: p.error('Capacity survival requires --fast for controlled body resets')
    if min(a.train_steps,a.eval_steps,a.initial_eval_steps)<=0: p.error('Step budgets must be positive')
    if a.worker: return worker(a)
    a.run_dir.mkdir(parents=True,exist_ok=False)
    write(a.run_dir/'status.json',dict(status='starting',started_at=time.time()))
    sources=[ROOT/'run_capacity_survival.py',ROOT/'grow_policy.py',ROOT/'sdaea_streaming_v2.py',ROOT/'sdaea_online_validate.py',ROOT/'fast_godot_env.py']
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
        flags=['--fixed-fps','20','--disable-vsync','--disable-render-loop'] if a.fast or a.uncapped else []
        if a.fast: flags+=['--transport=raw']
        flags += [f'--eye_width={a.eye_width}', f'--eye_height={a.eye_height}']
        game=subprocess.Popen([godot,*flags,'--path',str(ROOT.parent/'EnvolutionRobot'),'res://scenes/training_scene/capacity_survival.tscn',f'--port={a.port}','--env_seed=19'],stdout=gout,stderr=subprocess.STDOUT,env=env)
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
