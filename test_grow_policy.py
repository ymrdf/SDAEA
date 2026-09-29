import unittest
import torch
from sdaea_streaming_v2 import VisualNetwork,State,Learner,Config
from grow_policy import grow_network

class GrowthTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(19)
        self.source=VisualNetwork(27,27,256,3)
        self.state=State(torch.rand(1,6,96,96),torch.rand(1,6,6,8),torch.rand(1,30))
    def test_width_replication_preserves_function_before_noise(self):
        for width in (512,768):
            target=VisualNetwork(27,27,width,3)
            grow_network(target,self.source.state_dict(),256,3,noise=0)
            torch.testing.assert_close(target(self.state),self.source(self.state),rtol=2e-4,atol=2e-6)
    def test_added_layers_receive_updates(self):
        c=Config(width=512,depth=5,image_size=96)
        l=Learner(c,27,torch.device('cpu'))
        grow_network(l.actor,self.source.state_dict(),256,3)
        old=l.actor.hidden[12].weight.detach().clone()
        l.learn(self.state,0,1.,None,4,True)
        self.assertFalse(torch.equal(old,l.actor.hidden[12].weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in l.actor.parameters()))
    def test_replicas_not_identical_after_noise(self):
        target=VisualNetwork(27,27,512,3)
        grow_network(target,self.source.state_dict(),256,3)
        self.assertFalse(torch.equal(target.hidden[0].weight[:256],target.hidden[0].weight[256:]))
if __name__=='__main__': unittest.main()
