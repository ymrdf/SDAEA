import copy
import unittest
from dataclasses import replace
import torch
from sdaea_streaming_v2 import Config,Learner,State
from training_policy_batch import TrainingPolicyBatch

class TrainingBatchTests(unittest.TestCase):
    def test_independent_updates_match_serial(self):
        self.check_updates('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_batch_matches_serial_updates(self):
        self.check_updates('cuda')

    def check_updates(self, device):
        torch.set_num_threads(1);torch.manual_seed(72)
        c=Config(width=24,depth=3,image_size=24,recurrent_size=12,bptt_steps=3)
        configs=[c,replace(c,trace_steps=0,optimizer='adam',actor_lr=1e-4,critic_lr=3e-4),
                 replace(c,recurrent_size=0),replace(c,recurrent_size=0,actor_lr=.03)]
        serial=[Learner(x,2,torch.device(device)) for x in configs]
        batched=copy.deepcopy(serial);batch=TrainingPolicyBatch(batched)
        if device == 'cuda':
            for l in serial:
                if l.c.optimizer == 'adam':
                    for u in (l.actor_update,l.critic_update):
                        for group in u.optimizer.param_groups:
                            group['fused']=False;group['foreach']=True
        def state(dt):return State(torch.rand(1,6,24,24,device=device),torch.rand(1,6,6,8,device=device),torch.rand(1,5,device=device),dt)
        for step in range(7):
            states=[state(i+1) for i in range(4)]; actions=[0,1,0,1]
            terminal=[step==2,step==4,False,step==3]
            nxt=[None if terminal[i] else state(2) for i in range(4)]
            rewards=[.8,-.3,.2,-1.]
            probs=batch.prepare(states)
            for i,l in enumerate(serial):
                expected=l.prepare_action(states[i]).probs[0].detach()
                torch.testing.assert_close(torch.tensor(probs[i],dtype=expected.dtype,device=device),expected,atol=3e-6,rtol=3e-5)
            got=batch.learn(states,actions,rewards,nxt,2,terminal)
            for i,l in enumerate(serial):
                expected=l.learn(states[i],actions[i],rewards[i],nxt[i],2,terminal[i])
                self.assertAlmostEqual(got[i]['td'],expected['td'],places=5)
                for name in ('actor','critic'):
                    for k,v in getattr(l,name).state_dict().items():
                        torch.testing.assert_close(v,getattr(batched[i],name).state_dict()[k],atol=3e-6,rtol=3e-5)
                    if l.anchors[name] is not None:
                        torch.testing.assert_close(l.anchors[name],batched[i].anchors[name],atol=3e-6,rtol=3e-5)
                self.assertEqual(len(l.context),len(batched[i].context))
                for name in ('actor_update','critic_update'):
                    a,b=getattr(l,name),getattr(batched[i],name)
                    for x,y in zip(a.traces,b.traces):
                        torch.testing.assert_close(x,y,atol=3e-5,rtol=3e-4)
                    if l.c.optimizer=='adam':
                        for x,y in zip(a.parameters,b.parameters):
                            for key in ('exp_avg','exp_avg_sq'):
                                torch.testing.assert_close(a.optimizer.state[x][key],b.optimizer.state[y][key],atol=3e-6,rtol=3e-4)

if __name__=='__main__':unittest.main()
