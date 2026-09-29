import json,unittest
from types import SimpleNamespace
import numpy as np
from fast_godot_env import FastGodotEnv
from sdaea_online_validate import decode_eye
class Fragmented:
    def __init__(self,data): self.data=bytearray(data)
    def recv_into(self,view):
        n=min(len(view),31,len(self.data));view[:n]=self.data[:n];del self.data[:n];return n
class ProtocolTests(unittest.TestCase):
    def env(self,payload):
        e=FastGodotEnv.__new__(FastGodotEnv);e.connection=Fragmented(len(payload).to_bytes(4,'little')+payload);e.observation_spaces=[dict(left_eye=SimpleNamespace(shape=(3,300,320)))];return e
    def test_binary_rgb_lossless(self):
        pixels=np.arange(300*320*3,dtype=np.uint8).tobytes()
        header=json.dumps(dict(type='observe',physics_frame=12,obs=[dict(left_eye=dict(offset=0,length=len(pixels)),hp=[6])])).encode()
        obs=self.env(b'RGB1'+len(header).to_bytes(4,'little')+header+pixels)._get_json_dict()
        image=obs['obs'][0]['left_eye']
        self.assertEqual(image.tobytes(),pixels)
        np.testing.assert_array_equal(decode_eye(image,300,320,'eye'),decode_eye(pixels.hex(),300,320,'eye'))
    def test_negotiated_small_dimensions(self):
        pixels=bytes([70])*(150*160*3)
        header=json.dumps(dict(type='observe',physics_frame=12,obs=[dict(left_eye=dict(offset=0,length=len(pixels)))])).encode()
        e=self.env(b'RGB1'+len(header).to_bytes(4,'little')+header+pixels)
        e.observation_spaces=[dict(left_eye=SimpleNamespace(shape=(3,150,160)))]
        image=e._get_json_dict()['obs'][0]['left_eye']
        self.assertEqual(image.shape,(150,160,3))
        self.assertEqual(image.tobytes(),pixels)
    def test_hp_only_no_images(self):
        h=json.dumps(dict(type='step',physics_frame=13,obs=[dict(hp=[6])])).encode()
        e=self.env(b'RGB1'+len(h).to_bytes(4,'little')+h);e.physics_frame=12
        self.assertEqual(e._get_json_dict()['obs'],[dict(hp=[6])])
        self.assertEqual(getattr(e,'rgb_frames',0),0)
    def test_wrong_tick_rejected(self):
        h=json.dumps(dict(type='step',physics_frame=15,obs=[])).encode()
        e=self.env(b'RGB1'+len(h).to_bytes(4,'little')+h);e.physics_frame=12
        with self.assertRaises(RuntimeError): e._get_json_dict()
    def test_disconnect_not_infinite_loop(self):
        e=self.env(b'');e.connection=Fragmented(b'\x10\x00')
        with self.assertRaises(ConnectionError): e._get_json_dict()
    def test_legacy_json(self):
        self.assertEqual(self.env(b'{"type":"env_info"}')._get_json_dict(),dict(type='env_info'))
if __name__=='__main__': unittest.main()
