"""Backward-compatible JSON control plus framed raw RGB, with HP-only steps."""
import json
import numpy as np
from godot_rl.core.godot_env import GodotEnv

class FastGodotEnv(GodotEnv):
    def _recv_exact(self, length):
        result=bytearray(length);view=memoryview(result);offset=0
        while offset<length:
            n=self.connection.recv_into(view[offset:])
            if not n: raise ConnectionError('Godot disconnected during a frame')
            offset+=n
        return result

    def _get_json_dict(self):
        length=int.from_bytes(self._recv_exact(4),'little')
        if length>128*1024*1024: raise ValueError('Oversized observation frame')
        payload=self._recv_exact(length)
        self.bytes_received=getattr(self,'bytes_received',0)+length+4
        if payload[:4]!=b'RGB1': return json.loads(payload)
        size=int.from_bytes(payload[4:8],'little')
        if 8+size>len(payload): raise ValueError('Invalid RGB metadata length')
        response=json.loads(payload[8:8+size]);raw=memoryview(payload)[8+size:]
        for index,obs in enumerate(response['obs']):
            for key in ('left_eye','right_eye'):
                if key in obs:
                    offset,n=obs[key]['offset'],obs[key]['length']
                    channels,height,width=self.observation_spaces[index][key].shape
                    if channels!=3 or offset<0 or n!=height*width*channels or offset+n>len(raw):
                        raise ValueError('Invalid RGB payload bounds or size')
                    obs[key]=np.frombuffer(raw[offset:offset+n],dtype=np.uint8).reshape(height,width,channels)
        frame=response.get('physics_frame')
        previous=getattr(self,'physics_frame',None)
        if previous is not None and frame is not None:
            if response['type']=='step' and frame!=previous+1:
                raise RuntimeError(f'Expected one physics tick, got {frame-previous}')
            if response['type']=='observe' and frame!=previous:
                raise RuntimeError('Observation request advanced physics')
        self.physics_frame=frame
        if any('left_eye' in o for o in response['obs']):
            self.rgb_frames=getattr(self,'rgb_frames',0)+1
        return response

    def step(self, action, order_ij=False, include_rgb=True):
        action=self.action_space_processor.to_original_dist(action)
        self._send_as_json(dict(type='action',action=self.from_numpy(action,order_ij=order_ij),rgb=include_rgb))
        return self.step_recv()

    def observe(self):
        self._send_as_json(dict(type='observe'))
        response=self._get_json_dict()
        assert response['type']=='observe'
        return self._process_obs(response['obs'])
