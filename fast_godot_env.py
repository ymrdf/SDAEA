"""Backward-compatible JSON control plus framed raw RGB, with HP-only steps."""
import json
import os
import numpy as np
import socket
import stat
from gpu_vision_transport import GpuVisionTransport
from godot_rl.core.godot_env import GodotEnv

class FastGodotEnv(GodotEnv):
    def __init__(self, env_path=None, port=11008, show_window=False, seed=0,
                 framerate=None, action_repeat=None, speedup=None,
                 convert_action_space=False, transport='raw', **kwargs):
        self.transport = transport
        self.gpu_transport = None
        self.gpu_transports = {}
        self._gpu_server = None
        self._gpu_connection = None
        self._gpu_socket_path = None
        if transport == 'gpu':
            self._gpu_socket_path = kwargs.pop(
                'gpu_socket', f'/tmp/sdaea_gpu_{port}.sock')
            try:
                mode = os.stat(self._gpu_socket_path).st_mode
                if stat.S_ISSOCK(mode):
                    os.unlink(self._gpu_socket_path)
            except FileNotFoundError:
                pass
            self._gpu_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._gpu_server.bind(self._gpu_socket_path)
            self._gpu_server.listen(128)
            self._gpu_server.settimeout(GodotEnv.DEFAULT_TIMEOUT)
            kwargs['gpu_socket'] = self._gpu_socket_path
        kwargs['transport'] = transport
        super().__init__(
            env_path=env_path, port=port, show_window=show_window, seed=seed,
            framerate=framerate, action_repeat=action_repeat, speedup=speedup,
            convert_action_space=convert_action_space, **kwargs)

    def _accept_gpu_connection(self):
        if self.transport != 'gpu' or self.gpu_transport is not None:
            return
        try:
            for _ in range(self.num_envs):
                connection, _ = self._gpu_server.accept()
                with connection:
                    connection.settimeout(GodotEnv.DEFAULT_TIMEOUT)
                    transport = GpuVisionTransport.receive(connection)
                index = transport.agent_index
                if index not in range(self.num_envs) or index in self.gpu_transports:
                    transport.close()
                    raise ValueError(f'Duplicate or invalid GPU agent index: {index}')
                self.gpu_transports[index] = transport
            self.gpu_transport = self.gpu_transports[0]  # single-agent compatibility
        except BaseException:
            for transport in self.gpu_transports.values():
                transport.close()
            self.gpu_transports.clear()
            raise
        self._gpu_server.close()
        self._gpu_server = None
        try:
            os.unlink(self._gpu_socket_path)
        except FileNotFoundError:
            pass

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
        if payload[:4]!=b'RGB1':
            response=json.loads(payload)
            frame=response.get('physics_frame')
            previous=getattr(self,'physics_frame',None)
            if previous is not None and frame is not None:
                if response['type']=='step' and frame!=previous+1:
                    raise RuntimeError(f'Physics frame did not advance: {previous} -> {frame}')
                if response['type']=='observe' and frame!=previous:
                    raise RuntimeError('Observation request advanced physics')
            self.physics_frame=frame
            if response.get('obs') and any(o.get('gpu_rgb') for o in response['obs']):
                if self.gpu_transport is None:
                    raise RuntimeError('Godot sent GPU RGB without an active CUDA transport')
                for index, observation in enumerate(response['obs']):
                    if observation.get('gpu_rgb'):
                        frame_id=int(observation['gpu_frame_id'])
                        if int(observation.get('gpu_agent_index', -1)) != index:
                            raise ValueError('GPU observation agent order mismatch')
                        self.gpu_transports[index].attach_observation(observation,frame_id)
                        for eye in ('left', 'right'):
                            reference = observation.pop(f'gpu_reference_{eye}', None)
                            if reference is not None:
                                actual = observation[f'{eye}_eye'][..., :3].cpu().numpy().tobytes()
                                if actual != bytes.fromhex(reference):
                                    raise RuntimeError(f'GPU pixels differ from readback: agent {index}, {eye}, frame {frame_id}')
                                self.gpu_verified_eyes = getattr(self, 'gpu_verified_eyes', 0) + 1
                self.rgb_frames=getattr(self,'rgb_frames',0)+1
            return response
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
        self._accept_gpu_connection()
        response=self._get_json_dict()
        assert response['type']=='observe'
        return self._process_obs(response['obs'])

    def reset(self, seed=None):
        self._send_as_json(dict(type='reset'))
        self._accept_gpu_connection()
        response = self._get_json_dict()
        response['obs'] = self._process_obs(response['obs'])
        assert response['type'] == 'reset'
        return response['obs'], [{}] * self.num_envs

    def close(self):
        for transport in self.gpu_transports.values():
            transport.close()
        self.gpu_transports.clear()
        self.gpu_transport = None
        if self._gpu_connection is not None:
            self._gpu_connection.close()
            self._gpu_connection = None
        if self._gpu_server is not None:
            self._gpu_server.close()
            self._gpu_server = None
        try:
            super().close()
        finally:
            if self._gpu_socket_path:
                try:
                    os.unlink(self._gpu_socket_path)
                except FileNotFoundError:
                    pass
