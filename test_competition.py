import unittest
import torch
from run_competition import ensure_live
from sdaea_streaming_v2 import Config,Learner,State

class ResetEnv:
    def __init__(self): self.sent=[]
    def _send_as_json(self,message): self.sent.append(message)
    def _get_json_dict(self): return dict(type='reset',obs=[{'hp':[4.]},{'hp':[6.]}])
    def _process_obs(self,obs): return obs

class CompetitionTests(unittest.TestCase):
    def test_only_dead_agent_reset(self):
        env=ResetEnv()
        obs=ensure_live(env,[{'hp':[4.]},{'hp':[0.]}])
        self.assertEqual(env.sent,[dict(type='reset',agent_indices=[1])])
        self.assertEqual(obs[0]['hp'],[4.])
    def test_live_hp_change_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'changed live HP'):
            ensure_live(ResetEnv(),[{'hp':[3.]},{'hp':[0.]}])
    def test_deeper_network_learns(self):
        torch.set_num_threads(2)
        c=Config(depth=3,width=256,image_size=96)
        l=Learner(c,27,torch.device('cpu'))
        s=State(torch.rand(1,6,96,96),torch.rand(1,6,6,8),torch.rand(1,30))
        before=l.actor.hidden[6].weight.detach().clone()
        l.learn(s,0,1.,None,1,True)
        self.assertFalse(torch.equal(before,l.actor.hidden[6].weight))

if __name__=='__main__': unittest.main()
