#!/usr/bin/env python3
"""Compare actor kappa 3, 1, 0.3 in the no-block energy environment."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)

def source_hashes(project):
    paths = [ROOT / 'sdaea_streaming_v2.py', ROOT / 'sdaea_online_validate.py', ROOT / 'energy_policy.py', Path(__file__)]
    paths += [p for p in project.rglob('*') if p.is_file() and
              not any(part.startswith('.') for part in p.relative_to(project).parts) and
              p.suffix in ('.gd', '.tscn', '.godot', '.cs', '.csproj', '.glb')]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def summarize(directory):
    s = json.loads((directory / 'summary.json').read_text())
    with (directory / 'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    live = [r for r in rows if r['terminal'] == '0']
    config=json.loads((directory/'config.json').read_text())
    if not s['no_learn']:
        s['update_diagnostics'] = {
            key: {'mean':statistics.mean(float(r[key]) for r in rows),
                  'median':statistics.median(float(r[key]) for r in rows),
                  'max':max(float(r[key]) for r in rows)}
            for key in ('actor_lr','actor_trace_l1','actor_update_l1','critic_lr','td')}
        s['actor_bound_fraction'] = sum(float(r['actor_lr'])<config['actor_lr'] for r in rows)/len(rows)
        s['learning_windows'] = []
        for lower in (0,25000,50000,75000):
            # Split decisions at window boundaries using their elapsed environment steps.
            weighted=[(r,max(0,min(int(r['step']),lower+25000)-max(int(r['step'])-int(r['elapsed']),lower))) for r in rows]
            weighted=[(r,n) for r,n in weighted if n]
            nsteps=sum(n for r,n in weighted)
            idle=sum(n for r,n in weighted if all(json.loads(r['action'])[key]==1 for key in ('accelerate_forward','accelerate_sideways')))
            s['learning_windows'].append(dict(start=lower,steps=nsteps,no_acceleration_fraction=idle/nsteps))
        import torch
        before=torch.load(directory/'initial.pt',map_location='cpu',weights_only=True)
        after=torch.load(directory/'latest.pt',map_location='cpu',weights_only=True)
        s['relative_parameter_l2']={}
        for model in ('actor','critic'):
            for prefix in ('conv','hidden','output'):
                x=torch.cat([v.flatten() for k,v in before[model].items() if k.startswith(prefix)])
                y=torch.cat([v.flatten() for k,v in after[model].items() if k.startswith(prefix)])
                s['relative_parameter_l2'][model+'.'+prefix]=float((y-x).norm()/x.norm())

    steps = sum(int(r['elapsed']) for r in rows)
    actions = [(json.loads(r['action']), int(r['elapsed'])) for r in rows]
    s['no_acceleration_step_fraction'] = sum(n for a,n in actions if a['accelerate_forward']==1 and a['accelerate_sideways']==1)/steps
    s['hp_loss_per_step'] = -sum(float(r['hp_delta']) for r in rows)/steps
    s['hp_loss_expected_per_step'] = (1/180+(1-s['no_acceleration_step_fraction'])/90)/20
    s['energy_accounting_error'] = s['hp_loss_per_step']-s['hp_loss_expected_per_step']
    if any(int(r['positive_hp_events']) or int(r['negative_hp_events']) for r in rows):
        raise RuntimeError('Unexpected HP event in no-block experiment')
    if abs(s['energy_accounting_error']) > 2e-6:
        raise RuntimeError('HP loss does not match passive plus action energy cost')
    return dict(**s, positive_hp_events=sum(int(r['positive_hp_events']) for r in rows),
                negative_hp_events=sum(int(r['negative_hp_events']) for r in rows),
                mean_entropy=statistics.mean(float(r['entropy']) for r in rows),
                mean_max_probability=statistics.mean(float(r['max_probability']) for r in rows),
                identical_image_fraction=sum(float(r['image_change']) == 0 for r in live)/max(1,len(live)))

def check_frozen(reference, checkpoint):
    import torch
    before = torch.load(reference, map_location='cpu', weights_only=True)
    after = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for model in ('actor','critic'):
        if before[model].keys() != after[model].keys() or any(
            not torch.equal(t, after[model][k]) for k,t in before[model].items()):
            raise RuntimeError(f'Frozen evaluation changed {model} weights: {checkpoint}')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite-dir', type=Path, required=True)
    p.add_argument('--godot', type=Path, required=True)
    p.add_argument('--project', type=Path, default=ROOT.parent/'EnvolutionRobot')
    p.add_argument('--port', type=int, default=11048)
    a = p.parse_args()
    suite, project, godot = a.suite_dir.resolve(), a.project.resolve(), a.godot.resolve()
    if not godot.is_file() or not (project/'project.godot').is_file():
        p.error('Godot executable and project.godot must exist')
    suite.mkdir(parents=True, exist_ok=True)
    if any(suite.iterdir()):
        p.error('Use a new empty suite directory')
    (suite/'logs').mkdir()
    hashes = source_hashes(project)
    write_json(suite/'source_hashes.json',hashes)
    jobs = []
    for kappa, label in ((3.0, '3'), (1.0, '1'), (0.3, '0p3')):
        train_name = f'kappa_{label}_train_seed7'
        jobs.append(dict(name=train_name,group='train',seed=7,steps=100000,kappa=kappa,label=label))
        jobs.append(dict(name=f'kappa_{label}_eval_seed17',group='eval',seed=17,steps=24000,
                         kappa=kappa,label=label,reference=train_name))
    state = dict(status='running',started_at=time.time(),total_environment_steps=372000,
                 completed=[],jobs=jobs,notes='Only actor_kappa varies: 3, 1, 0.3. Fresh training seed 7 and frozen evaluation seed 17 for every arm; single training seed diagnostic, not multi-seed evidence. No blocks; original HP costs, gamma, traces, entropy and architecture.')
    write_json(suite/'status.json',state)
    write_json(suite/'plan.json',dict(jobs=jobs,device='cuda',hold_steps=4,port=a.port,project=str(project),
                                    notes=state['notes']))
    trainer = game = None
    def stop(_sig,_frame):
        raise KeyboardInterrupt('Suite interrupted')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        for job in jobs:
            if source_hashes(project) != hashes:
                raise RuntimeError('Training or environment source changed; stop to preserve comparability')
            state.update(current_job=job['name'],job_started_at=time.time())
            write_json(suite/'status.json',state)
            run_dir = suite/job['name']
            command = [sys.executable,'-u',str(ROOT/'energy_policy.py'),'--run-dir',str(run_dir),
                       '--seed',str(job['seed']),'--max-steps',str(job['steps']),'--device','cuda',
                       '--port',str(a.port),'--actor-kappa',str(job['kappa'])]
            reference = None
            if job['group'] == 'eval':
                reference = suite/job['reference']/'latest.pt'
                command += ['--no-learn','--warm-start',str(reference)]
            trainer_log = suite/'logs'/f"{job['name']}.python.log"
            game_log = suite/'logs'/f"{job['name']}.godot.log"
            print('Starting',job['name'],flush=True)
            with trainer_log.open('w') as out, game_log.open('w') as gout:
                trainer = subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+90
                while 'waiting for remote GODOT connection' not in trainer_log.read_text():
                    if trainer.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError(f"Trainer did not open listener: {trainer_log}")
                    time.sleep(.25)
                game = subprocess.Popen([str(godot),'--path',str(project),'res://scenes/training_scene/energy_scene.tscn',f'--port={a.port}',
                                         f"--env_seed={job['seed']}"],stdout=gout,stderr=subprocess.STDOUT,start_new_session=True)
                last_progress=time.monotonic()
                previous_size=-1
                while trainer.poll() is None:
                    time.sleep(2)
                    metrics=run_dir/'metrics.csv'
                    size=metrics.stat().st_size if metrics.exists() else 0
                    if size != previous_size:
                        previous_size=size
                        last_progress=time.monotonic()
                    if game.poll() is not None:
                        # Allow the trainer to finish saving after its normal close handshake.
                        try: trainer.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            raise RuntimeError(f'Godot exited before trainer: {game_log}')
                    if time.monotonic()-last_progress>600:
                        raise RuntimeError(f'No metrics progress for ten minutes: {trainer_log}')
                if trainer.returncode != 0:
                    raise RuntimeError(f'Trainer exited {trainer.returncode}: {trainer_log}')
                game.wait(timeout=30)
                if game.returncode != 0:
                    raise RuntimeError(f'Godot exited {game.returncode}: {game_log}')
            result=summarize(run_dir)
            if result['steps'] != job['steps']:
                raise RuntimeError(f"Run stopped early at {result['steps']}: {run_dir}")
            if job['group'] != 'train':
                if result['updates'] != 0:
                    raise RuntimeError(f'Evaluation performed learning updates: {run_dir}')
                check_frozen(reference or run_dir/'initial.pt',run_dir/'latest.pt')
            result.update(group=job['group'],seed=job['seed'],name=job['name'],kappa=job['kappa'],label=job['label'])
            result['frozen_weights_verified'] = job['group'] == 'eval'
            if job['group']=='train':
                if state['completed']:
                    check_frozen(suite/'kappa_3_train_seed7'/'initial.pt',run_dir/'initial.pt')
                result['initial_weights_matched'] = True
            state['completed'].append(result)
            write_json(suite/'status.json',state)
            trainer=game=None
        groups={}
        for label in ('3','1','0p3'):
            group_runs=[r for r in state['completed'] if r['label']==label]
            groups[label]={r['group']:r for r in group_runs}
        write_json(suite/'comparison.json',dict(groups=groups,runs=state['completed'],notes=state['notes']))
        state.update(status='complete',finished_at=time.time(),current_job=None)
        write_json(suite/'status.json',state)
        print('Comparison complete:',suite/'comparison.json',flush=True)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),finished_at=time.time())
        write_json(suite/'status.json',state)
        raise
    finally:
        if trainer is not None and trainer.poll() is None:
            trainer.send_signal(signal.SIGINT)
            try: trainer.wait(timeout=30)
            except subprocess.TimeoutExpired:
                trainer.kill(); trainer.wait()
        if game is not None and game.poll() is None:
            game.terminate()
            try: game.wait(timeout=10)
            except subprocess.TimeoutExpired:
                game.kill(); game.wait()

if __name__=='__main__':
    main()
