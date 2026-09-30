import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import torch
from sdaea_streaming_v2 import Config,Learner,State,validate
from frozen_policy_batch import FrozenPolicyBatch

class LongTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(81)
        self.c=Config(width=24,depth=3,image_size=24,recurrent_size=12)
    def state(self,elapsed=4):
        return State(torch.rand(1,6,24,24),torch.rand(1,6,6,8),torch.rand(1,5),elapsed)

    def test_chunk_plan_has_unique_checkpoints_and_exact_training_budget(self):
        from types import SimpleNamespace
        from run_survival_search import phase_plan
        phases=phase_plan(SimpleNamespace(initial_eval_steps=5000,train_steps=300000,
            train_chunk_steps=100000,intermediate_eval_steps=20000,eval_steps=50000))
        self.assertEqual(sum(n for name,n in phases if name.startswith('train_')),300000)
        self.assertEqual(sum(n for _,n in phases),395000)
        self.assertEqual(len({name for name,_ in phases}),len(phases))
        self.assertEqual(phases[-1],['trained',50000])

    def test_prepared_actor_matches_recomputed_update(self):
        a=Learner(self.c,2,torch.device('cpu'));b=Learner(self.c,2,torch.device('cpu'))
        b.actor.load_state_dict(a.actor.state_dict());b.critic.load_state_dict(a.critic.state_dict())
        for j in range(6):
            s,nxt=self.state(),self.state(1)
            a.prepare_action(s)
            a.learn(s,j%2,.5,nxt,1,False);b.learn(s,j%2,.5,nxt,1,False)
        for model in ('actor','critic'):
            for k,v in getattr(a,model).state_dict().items():
                torch.testing.assert_close(v,getattr(b,model).state_dict()[k])
        self.assertIsNone(a.pending_actor)

    def test_frozen_batch_matches_serial_with_mixed_elapsed_and_resets(self):
        configs=[replace(self.c,no_learn=True),replace(self.c,no_learn=True),
                 replace(self.c,no_learn=True,recurrent_size=0)]
        learners=[Learner(c,2,torch.device('cpu')) for c in configs]
        variants=[{},dict(clear_memory=True),{}]
        batch=FrozenPolicyBatch(learners,variants,range(3))
        for j in range(5):
            states=[self.state(0 if j==0 else i+1) for i in range(3)]
            learners[1].clear()
            result=batch.probabilities(states)
            for i,l in enumerate(learners):
                with torch.no_grad():expected=l.distribution(states[i]).probs[0]
                torch.testing.assert_close(torch.tensor(result[i],dtype=expected.dtype),expected,atol=2e-6,rtol=2e-5)
                l.learn(states[i],0,0.,self.state(),4,False)
            if j==2:
                batch.clear_indices([0]);learners[0].clear()

    def test_adam_changes_vision_and_restores_optimizer(self):
        c=replace(self.c,recurrent_size=0,optimizer='adam',trace_steps=0,actor_lr=1e-4,critic_lr=3e-4)
        validate(c);a=Learner(c,2,torch.device('cpu'));s=self.state()
        before=a.actor.conv[0].weight.detach().clone()
        a.learn(s,0,1.,None,1,True)
        self.assertFalse(torch.equal(before,a.actor.conv[0].weight))
        self.assertTrue(a.actor_update.optimizer.state)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.pt';a.save(path,[],1)
            b=Learner(c,2,torch.device('cpu'));b.load(path,[])
            a.learn(s,1,-.4,None,1,True);b.learn(s,1,-.4,None,1,True)
            for k,v in a.actor.state_dict().items():torch.testing.assert_close(v,b.actor.state_dict()[k])
            old=Learner(replace(c,optimizer='bounded'),2,torch.device('cpu'))
            with self.assertRaisesRegex(ValueError,'optimizer'):old.load(path,[])
            old.load(path,[],allow_optimizer_change=True)

if __name__=='__main__':unittest.main()
