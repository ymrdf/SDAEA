import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import torch
from sdaea_streaming_v2 import Config, Learner, body_signal

class BodyObjectiveTests(unittest.TestCase):
    def test_linear_hp_signal_and_terminal_do_not_use_reset_hp(self):
        c=Config(energy_delta=1.,alive=0.,shaping=0.,death_cost=0.)
        self.assertEqual(body_signal(6,11,False,c)['valence'],1.)
        self.assertEqual(body_signal(6,1,False,c)['valence'],-1.)
        terminal=body_signal(6,100,True,c)
        self.assertEqual(terminal['pleasure'],0.)
        self.assertEqual(terminal['valence'],-1.2)
        self.assertEqual(terminal['valence'],terminal['pleasure']-terminal['pain'])

    def test_objective_transfer_requires_explicit_flag(self):
        c=Config(width=24)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.pt'
            source=Learner(c,2,torch.device('cpu'));source.save(path,[],0)
            target=Learner(replace(c,gamma=.9999,energy_delta=1.),2,torch.device('cpu'))
            with self.assertRaises(ValueError):target.load(path,[])
            target.load(path,[],allow_objective_change=True)
            for k,v in source.actor.state_dict().items():
                self.assertTrue(torch.equal(v,target.actor.state_dict()[k]))

if __name__=='__main__':unittest.main()
