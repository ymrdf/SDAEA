#!/usr/bin/env python3
"""Capacity expansion and continued learning in one shared competitive world."""
import argparse, csv, hashlib, json, os, shutil, signal, subprocess, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from sdaea_streaming_v2 import Config, Learner, SensoryMemory, State, body_signal, motor_table
from sdaea_online_validate import ActionSpec, extract_hp, seed_everything

VARIANTS = []
LABELS = []
N = 0
FROZEN = set()
RANDOM = set()
STATIONARY = set()
ROOT=Path(__file__).resolve().parent

class PolicyMemory(SensoryMemory):
    """Ablate both current pixels and the visual EMA; body input is unchanged."""
    def __init__(self, *args, vision_ablation=None):
        super().__init__(*args)
        if vision_ablation not in (None, 'gray'):
            raise ValueError('Unknown vision ablation')
        self.vision_ablation = vision_ablation

    def state(self, observation, previous_action, elapsed):
        state = super().state(observation, previous_action, elapsed)
        if self.vision_ablation == 'gray':
            return State(torch.full_like(state.image, .5), torch.full_like(state.history, .5),
                         state.body, state.elapsed)
        return state


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

def phase_plan(a):
    result=[]
    if a.initial_eval_steps:
        result.append(['initial',a.initial_eval_steps])
    if a.train_chunk_steps and a.train_steps:
        remaining=a.train_steps;index=0
        while remaining:
            index+=1;steps=min(a.train_chunk_steps,remaining);remaining-=steps
            result.append([f'train_{index}',steps])
            if remaining and a.intermediate_eval_steps:
                result.append([f'eval_{index}',a.intermediate_eval_steps])
    elif a.train_steps:
        result.append(['train',a.train_steps])
    result.append(['trained',a.eval_steps])
    return result


def worker(a):
    from godot_rl.core.godot_env import GodotEnv
    torch.set_num_threads(2)
    if a.fast:
        from fast_godot_env import FastGodotEnv
        GodotEnv=FastGodotEnv
    env=GodotEnv(env_path=None,port=a.port,show_window=True,seed=a.seed)
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
    for i, variant in enumerate(VARIANTS):
        seed_everything(a.seed + i)
        options = dict(image_size=96, width=256, depth=3)
        options.update(variant.get('config', {}))
        c=Config(**options, eye_height=a.eye_height,eye_width=a.eye_width,device='cuda',seed=a.seed+i,
                 no_learn=True,random_actions=i in RANDOM)
        from sdaea_streaming_v2 import validate
        validate(c)
        l=Learner(c,len(table),torch.device('cuda'))
        learners.append(l);configs.append(c);memories.append(PolicyMemory(c,torch.device('cuda'),len(table),vision_ablation=variant.get('vision_ablation')))
        if variant.get('checkpoint'):
            l.load(Path(variant['checkpoint']), specs, allow_objective_change=variant.get('transfer_objective',False), allow_optimizer_change=variant.get('transfer_optimizer',False))
        elif i not in RANDOM | STATIONARY:
            # Transfer matching feedforward paths; extra CNN/memory parameters are new.
            for name in ('actor','critic'):
                model=getattr(l,name)
                if c.width!=256:
                    grow_network(model,source[name],256,3)
                else:
                    own=model.state_dict()
                    for key,value in source[name].items():
                        if key in own and own[key].shape==value.shape:
                            own[key].copy_(value)
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
              phases=phase_plan(a),
              source_checkpoint=str(a.source_checkpoint),shared_scene=True,fast=a.fast,
              parameter_counts={label:sum(p.numel() for model in (l.actor,l.critic) for p in model.parameters()) for label,l in zip(LABELS,learners)},
              variants=VARIANTS, seed=a.seed, green_blocks=a.green_blocks, red_blocks=a.red_blocks,
              notes='All policies compete simultaneously in one scene. New memory/CNN connections change the initial policy; matching feedforward weights transferred from source. Equal HP6 at each phase; resources/world continue evolving. Single training seed per variant. Initial and final evaluation weights are frozen. No block labels or environment rewards enter policy or learning.')
    write(a.run_dir/'plan.json',plan)
    total=0;results={};timings={};stop=False;reset_checks=0;visual_changes=[0]*N
    def interrupt(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    fields=['step','phase','agent','elapsed','hp','terminal','valence','positive_hp_events','negative_hp_events','updates','action','green_blocks','red_blocks','td','entropy','max_probability','actor_lr','actor_trace_l1','actor_update_l1']
    try:
      with (a.run_dir/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for phase,length in plan['phases']:
          is_training=phase=='train' or phase.startswith('train_')
          for i,l in enumerate(learners):
            l.c.no_learn=not is_training or i in FROZEN;l.clear()
          # Equal body energy at the start of every phase, including frozen evaluation.
          frame=getattr(env,'physics_frame',None)
          assert all(env.call('restart_competition_body'))
          obs=env.observe()
          assert frame==env.physics_frame
          assert all(abs(extract_hp(o,'hp')-6.)<1e-6 for o in obs)
          for i in range(N):
            memories[i].clear();states[i]=memories[i].state(obs[i],None,0)
          frozen_batch=None
          if a.batch_eval and not is_training:
            from frozen_policy_batch import FrozenPolicyBatch
            frozen_batch=FrozenPolicyBatch(learners,VARIANTS,[i for i in range(N) if i not in RANDOM | STATIONARY])
          lives=[0]*N;lifetimes=[[] for _ in range(N)];first_death=[None]*N
          frozen=[{k:v.detach().clone() for k,v in l.actor.state_dict().items()} for l in learners]
          frozen_critics=[{k:v.detach().clone() for k,v in l.critic.state_dict().items()} for l in learners]
          stats=[dict(steps=0,deaths=0,positive_hp_events=0,negative_hp_events=0,hp_sum=0.,updates=0,no_acceleration_steps=0) for _ in LABELS]
          start_counts=[np.asarray(o['block_counts']).copy() for o in obs]
          phase_started=time.perf_counter();phase_bytes=getattr(env,'bytes_received',0);phase_frames=getattr(env,'rgb_frames',0)
          stage_times=dict(action=0.,environment=0.,learning_and_logging=0.)
          elapsed_phase=0
          write(a.run_dir/'status.json',dict(status='running',phase=phase,step=total,phase_step=0,total_steps=sum(length for _,length in plan['phases']),updated_at=time.time()))
          while elapsed_phase<length and not stop:
            stage_start=time.perf_counter()
            actions=[]
            batch_probs={} if frozen_batch is None else frozen_batch.probabilities(states)
            for i,l in enumerate(learners):
              if VARIANTS[i].get('clear_memory'):
                if not l.c.no_learn: raise ValueError('Memory ablation is evaluation-only')
                l.clear()
              if i in STATIONARY:
                actions.append(table.index([1 if s.learned else s.fixed_value for s in specs]))
              elif i in RANDOM: actions.append(int(rngs[i].integers(len(table))))
              elif frozen_batch is not None:
                probs=batch_probs[i]
                actions.append(int(rngs[i].choice(len(table),p=probs)))
              else:
                probs=l.prepare_action(states[i]).probs[0].detach().cpu().numpy().astype(float)
                actions.append(int(rngs[i].choice(len(table),p=probs/probs.sum())))
            stage_times['action']+=time.perf_counter()-stage_start
            stage_start=time.perf_counter()
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
            stage_times['environment']+=time.perf_counter()-stage_start
            stage_start=time.perf_counter()
            for i,l in enumerate(learners):
              nxt=None if term[i] else memories[i].state(obs[i],actions[i],elapsed)
              if a.verify_visuals and nxt is not None:
                visual_changes[i]+=int(not torch.equal(states[i].image,nxt.image))
              update_metrics={}
              if frozen_batch is not None and i in batch_probs:
                probs=batch_probs[i]
                update_metrics=dict(entropy=float(-(probs*np.log(probs)).sum()),max_probability=float(probs.max()))
              elif i not in RANDOM | STATIONARY:
                update_metrics=l.learn(states[i],actions[i],values[i],nxt,elapsed,term[i])
              lives[i]+=elapsed
              if term[i]:
                lifetimes[i].append(lives[i]);lives[i]=0
                if first_death[i] is None: first_death[i]=elapsed_phase
              action_dict=dict(zip([s.name for s in specs],table[actions[i]]))
              stats[i]['no_acceleration_steps']+=elapsed*int(action_dict['accelerate_forward']==1 and action_dict['accelerate_sideways']==1)
              t=stats[i];t['steps']+=elapsed;t['deaths']+=int(term[i]);t['positive_hp_events']+=pos[i];t['negative_hp_events']+=neg[i];t['updates']+=int(not l.c.no_learn)
              w.writerow(dict(step=total,phase=phase,agent=LABELS[i],elapsed=elapsed,hp=previous[i],terminal=int(term[i]),valence=values[i],positive_hp_events=pos[i],negative_hp_events=neg[i],updates=t['updates'],action=actions[i],green_blocks=int(obs[i]['block_counts'][0]),red_blocks=int(obs[i]['block_counts'][1]),**{key:update_metrics.get(key) for key in ('td','entropy','max_probability','actor_lr','actor_trace_l1','actor_update_l1')}))
              states[i]=nxt
            stage_times['learning_and_logging']+=time.perf_counter()-stage_start
            if any(term):
              if frozen_batch is not None: frozen_batch.clear_indices([i for i,t in enumerate(term) if t])
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
                  interval=states[i].elapsed
                  states[i]=memories[i].state(reset_obs[i],actions[i],0)
                  states[i].elapsed=interval
              obs=reset_obs
            if total//500 != (total-elapsed)//500:
              f.flush();write(a.run_dir/'status.json',dict(status='running',phase=phase,step=total,phase_step=elapsed_phase,total_steps=sum(length for _,length in plan['phases']),updated_at=time.time(),phase_seconds=time.perf_counter()-phase_started,shared_steps_per_second=elapsed_phase/(time.perf_counter()-phase_started),cuda_allocated_mb=torch.cuda.memory_allocated()/2**20,cuda_peak_mb=torch.cuda.max_memory_allocated()/2**20))
              print(f'phase={phase} shared_step={total}',flush=True)
            if total//10000 != (total-elapsed)//10000:
              for i,l in enumerate(learners): l.save(a.run_dir/LABELS[i]/'latest.pt',specs,total)
          if stop: raise KeyboardInterrupt('Stopped')
          for i,l in enumerate(learners):
            if not is_training or i in FROZEN:
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
          timings[phase]=dict(stage_wall_seconds=stage_times,steps=length,seconds=time.perf_counter()-phase_started,bytes_received=getattr(env,'bytes_received',0)-phase_bytes,rgb_observations=getattr(env,'rgb_frames',0)-phase_frames)
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
    p=argparse.ArgumentParser();p.add_argument('--batch-eval',action='store_true');p.add_argument('--train-chunk-steps',type=int,default=0);p.add_argument('--intermediate-eval-steps',type=int,default=20000);p.add_argument('--variants',type=Path,required=True);p.add_argument('--seed',type=int,default=29);p.add_argument('--green-blocks',type=int,default=60);p.add_argument('--red-blocks',type=int,default=200);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--port',type=int,default=11088);p.add_argument('--train-steps',type=int,default=50000);p.add_argument('--eval-steps',type=int,default=50000);p.add_argument('--worker',action='store_true');p.add_argument('--fast',action='store_true');p.add_argument('--uncapped',action='store_true');p.add_argument('--source-checkpoint',type=Path,required=True);p.add_argument('--initial-eval-steps',type=int,default=10000);p.add_argument('--verify-visuals',action='store_true');p.add_argument('--eye-width',type=int,default=160);p.add_argument('--eye-height',type=int,default=150);a=p.parse_args()
    global VARIANTS, LABELS, N, FROZEN, RANDOM, STATIONARY
    VARIANTS=json.loads(a.variants.read_text())
    LABELS=[v['label'] for v in VARIANTS];N=len(LABELS)
    if len(set(LABELS))!=N or any(not label.replace('_','').isalnum() for label in LABELS):
        p.error('Labels must be unique safe names')
    FROZEN={i for i,v in enumerate(VARIANTS) if v.get('role') in ('frozen','random','stationary')}
    RANDOM={i for i,v in enumerate(VARIANTS) if v.get('role')=='random'}
    STATIONARY={i for i,v in enumerate(VARIANTS) if v.get('role')=='stationary'}
    if not RANDOM or not FROZEN or min(a.green_blocks,a.red_blocks)<0:
        p.error('Require controls and nonnegative block counts')
    if min(a.train_chunk_steps,a.intermediate_eval_steps)<0:p.error('Chunk budgets must be nonnegative')
    if min(a.eye_width,a.eye_height)<36: p.error('Eye dimensions must be >=36')
    if not a.fast: p.error('Capacity survival requires --fast for controlled body resets')
    if min(a.train_steps,a.initial_eval_steps)<0 or a.eval_steps<=0: p.error('Require nonnegative training/initial budgets and positive final evaluation')
    if a.worker: return worker(a)
    a.run_dir.mkdir(parents=True,exist_ok=False)
    write(a.run_dir/'status.json',dict(status='starting',started_at=time.time()))
    scene=ROOT.parent/'EnvolutionRobot'/'scenes/training_scene'/f'search_{a.run_dir.name}.tscn'
    scene.write_text('[gd_scene load_steps=3 format=3]\n[ext_resource type="PackedScene" path="res://scenes/training_scene/training_scene.tscn" id="1"]\n[ext_resource type="Script" path="res://scenes/training_scene/competition.gd" id="2"]\n[node name="Competition" instance=ExtResource("1")]\nscript = ExtResource("2")\n[node name="PlayingArea" parent="." index="2"]\n'+f'number_of_robots_to_spawn = {N}\npolicy_labels = Array[String]({json.dumps(LABELS)})\nnumber_of_green_blocks_to_spawn = {a.green_blocks}\nnumber_of_red_blocks_to_spawn = {a.red_blocks}\nmin_green_blocks = {a.green_blocks}\nmin_red_blocks = {a.red_blocks}\n')
    shutil.copy2(a.variants,a.run_dir/'variants.json')
    sources=[ROOT/'run_survival_search.py',ROOT/'frozen_policy_batch.py',ROOT/'grow_policy.py',ROOT/'sdaea_streaming_v2.py',ROOT/'sdaea_online_validate.py',ROOT/'fast_godot_env.py']
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
        game=subprocess.Popen([godot,*flags,'--path',str(ROOT.parent/'EnvolutionRobot'),f'res://scenes/training_scene/{scene.name}',f'--port={a.port}',f'--env_seed={a.seed}'],stdout=gout,stderr=subprocess.STDOUT,env=env)
        last=time.monotonic();size=-1;last_gpu=0
        gpu_log=a.run_dir/'gpu.csv'
        gpu_log.write_text('timestamp,utilization_gpu,utilization_memory,memory_used,power_draw\n')
        while trainer.poll() is None:
          time.sleep(2)
          if time.monotonic()-last_gpu>=20:
            sample=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            with gpu_log.open('a') as g: g.write(sample.stdout)
            last_gpu=time.monotonic()
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
