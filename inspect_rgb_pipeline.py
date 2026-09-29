"""Capture received observations from a fresh seven-robot scene without learning."""
import csv,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from fast_godot_env import FastGodotEnv
from sdaea_online_validate import observation_to_tensor
from run_competition import ensure_live,reset_indices

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/rgb_inspection_20260929'

def worker():
    env=FastGodotEnv(port=11099,show_window=True,seed=19)
    records=[];captured={};checks=0;changed=[0]*14
    try:
        obs=ensure_live(env,env.reset()[0])
        actions=[]
        for space in env.action_spaces:
            controls=dict(accelerate_forward=1,accelerate_sideways=1,shoot=0,turn=2)
            actions.append([controls[k] for k in space])
        def capture(step,obs):
            nonlocal checks
            legacy=env.call('get_obs')
            for i,o in enumerate(obs):
                for eye in ('left_eye','right_eye'):
                    arr=o[eye]
                    assert arr.shape==(150,160,3) and arr.dtype==np.uint8
                    assert arr.tobytes()==bytes.fromhex(legacy[i][eye])
                    Image.fromarray(arr).save(OUT/f'step{step:03d}_agent{i}_{eye}.png')
                    checks+=1
            captured[step]=[{k:np.array(v,copy=True) for k,v in o.items()} for o in obs]
        capture(0,obs)
        previous=obs
        for step in range(1,65):
            rgb=step%4==0
            obs,_,done,trunc,_=env.step(actions,order_ij=True,include_rgb=rgb)
            assert all(('left_eye' in o)==rgb and ('right_eye' in o)==rgb for o in obs)
            hp_before=[o['hp'][0] for o in obs]
            if not rgb:
                # Obtain an explicit snapshot at selected boundaries only; no hidden tick.
                if step==1:
                    frame=env.physics_frame
                    shot=env.observe()
                    assert env.physics_frame==frame and hp_before==[o['hp'][0] for o in shot]
            if rgb:
                for i in range(7):
                    for j,eye in enumerate(('left_eye','right_eye')):
                        changed[2*i+j]+=int(np.any(obs[i][eye]!=previous[i][eye]))
                previous=obs
            dead=[i for i,o in enumerate(obs) if o['hp'][0]<=0 or done[i] or trunc[i]]
            if dead:
                obs=ensure_live(env,reset_indices(env,dead));previous=obs
            if step in (16,32,64):
                if 'left_eye' not in obs[0]: obs=env.observe()
                capture(step,obs)
            records.append(dict(step=step,rgb_requested=rgb,physics_frame=env.physics_frame,hp=hp_before))
        assert all(n>0 for n in changed)
        # Save the actual preprocessing output (not a PIL approximation).
        sample=captured[32][5]
        for size in (48,96,144):
            tensor=observation_to_tensor(sample,'left_eye','right_eye',150,160,size,torch.device('cpu'))
            assert tensor.shape==(1,6,size,size) and 0<=tensor.min()<=tensor.max()<=1
            for eye,j in [('left_eye',0),('right_eye',3)]:
                arr=(tensor[0,j:j+3].permute(1,2,0).numpy()*255).round().astype(np.uint8)
                Image.fromarray(arr).save(OUT/f'model_{size}_{eye}.png')
        labels=env.agent_policy_names
        # Grid of every robot, with original eye images enlarged 2x, no enhancement.
        sheet=Image.new('RGB',(700,7*335),(24,27,32));draw=ImageDraw.Draw(sheet)
        for i,label in enumerate(labels):
            y=i*335;draw.text((12,y+8),f'Agent {i} | {label} | step 32 | LEFT / RIGHT',fill='white')
            for j,eye in enumerate(('left_eye','right_eye')):
                tile=Image.fromarray(captured[32][i][eye]).resize((320,300),Image.Resampling.NEAREST)
                sheet.paste(tile,(12+j*344,y+28))
        sheet.save(OUT/'all_agents.png')
        # User-facing detail: one camera pair and the exact network inputs.
        sheet=Image.new('RGB',(1020,820),(24,27,32));draw=ImageDraw.Draw(sheet)
        draw.text((15,10),'Received RGB | agent 5: 96_w256_d3 | step 32 | 160x150 per eye',fill='white')
        for j,eye in enumerate(('left_eye','right_eye')):
            sheet.paste(Image.fromarray(sample[eye]).resize((480,450),Image.Resampling.NEAREST),(15+j*510,40))
            draw.text((15+j*510,25),eye,fill='white')
        draw.text((15,515),'Actual model input, left eye | 48x48 / 96x96 / 144x144 (nearest enlarged)',fill='white')
        for j,size in enumerate((48,96,144)):
            im=Image.open(OUT/f'model_{size}_left_eye.png').resize((240,240),Image.Resampling.NEAREST)
            sheet.paste(im,(15+j*340,560));draw.text((15+j*340,540),str(size),fill='white')
        sheet.save(OUT/'received_and_model_inputs.png')
        # Four time samples of the same received left eye.
        seq=Image.new('RGB',(680,680),(24,27,32));d=ImageDraw.Draw(seq)
        for idx,step in enumerate((0,16,32,64)):
            x=(idx%2)*340;y=(idx//2)*340
            d.text((x+10,y+8),f'agent 5 left | step {step}',fill='white')
            seq.paste(Image.fromarray(captured[step][5]['left_eye']).resize((320,300),Image.Resampling.NEAREST),(x+10,y+28))
        seq.save(OUT/'turning_sequence.png')
        result=dict(status='complete',scene='fresh shared competition scene; scripted turn-in-place, no model updates',eye_shape=[150,160,3],dtype='uint8',range=[0,255],raw_hex_exact_matches=checks,changed_counts=changed,physics_steps=64,all_rgb_steps_checked=True,labels=labels,records=records)
        (OUT/'report.json').write_text(json.dumps(result,indent=2))
    finally: env.close()

if __name__=='__main__':
    if '--worker' in sys.argv: worker();sys.exit()
    OUT.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['DOTNET_ROOT']='/home/gao/Documents/Codex/2026-09-23-godot-net/.tools/dotnet';env['PATH']=env['DOTNET_ROOT']+':'+env['PATH']
    game=None;child=None
    with (OUT/'python.log').open('w') as log,(OUT/'godot.log').open('w') as gout:
      try:
        child=subprocess.Popen([sys.executable,'-u',__file__,'--worker'],stdout=log,stderr=subprocess.STDOUT,env=env)
        deadline=time.monotonic()+60
        while 'waiting for remote GODOT connection' not in (OUT/'python.log').read_text():
            if child.poll() is not None or time.monotonic()>deadline: raise RuntimeError('Listener failed')
            time.sleep(.1)
        godot='/home/gao/Documents/Codex/2026-09-23-godot-net/.tools/godot/Godot_v4.7.2-stable_mono_linux_x86_64/Godot_v4.7.2-stable_mono_linux.x86_64'
        game=subprocess.Popen([godot,'--fixed-fps','20','--disable-vsync','--disable-render-loop','--transport=raw','--eye_width=160','--eye_height=150','--path',str(ROOT.parent/'EnvolutionRobot'),'res://scenes/training_scene/competition.tscn','--port=11099','--env_seed=19'],stdout=gout,stderr=subprocess.STDOUT,env=env)
        child.wait(timeout=120)
        if child.returncode: raise RuntimeError((OUT/'python.log').read_text()[-2000:])
        print(OUT)
      finally:
        for p in (child,game):
            if p is not None and p.poll() is None:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
