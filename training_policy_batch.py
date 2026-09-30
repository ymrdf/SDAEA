"""Batch independent learners' forward/backward passes, preserving their optimizers.

Stacking parameters is differentiable: gradients flow to each original parameter.
No weights, optimizer moments, recurrent anchors or eligibility traces are shared.
Group by architecture and history length so deaths never pad/change BPTT semantics.
"""
from collections import defaultdict
import torch
from torch import nn
from torch.func import functional_call, vmap
from torch.distributions import Categorical
from sdaea_streaming_v2 import State


class Window(nn.Module):
    def __init__(self, model, recurrent, sequence):
        super().__init__(); self.model=model; self.recurrent=recurrent; self.sequence=sequence

    def forward(self, image, history, body, hidden, elapsed):
        state=State(image,history,body,elapsed[0])
        if not self.recurrent:
            return self.model(state), hidden, hidden
        if self.sequence:
            return self.model.sequence(self.model.visual_features(state),hidden,elapsed)
        output,nxt=self.model(state,hidden)
        return output,nxt,nxt


def evaluate(learners, indices, name, sequences, anchors, sequence=True):
    models=[getattr(learners[i],name) for i in indices]
    parameter_sets=[dict(m.named_parameters()) for m in models]
    buffer_sets=[dict(m.named_buffers()) for m in models]
    params={f'model.{k}':torch.stack([p[k] for p in parameter_sets]) for k in parameter_sets[0]}
    buffers={f'model.{k}':torch.stack([b[k] for b in buffer_sets]) for k in buffer_sets[0]}
    sample=next(models[0].parameters()); size=learners[indices[0]].c.recurrent_size
    hidden=torch.stack([sample.new_zeros(1,size or 1) if h is None else h for h in anchors])
    args=[torch.stack([torch.cat([getattr(s,key) for s in seq]) for seq in sequences])
          for key in ('image','history','body')]
    elapsed=sample.new_tensor([[s.elapsed for s in seq] for seq in sequences])
    wrapper=Window(models[0],bool(size),sequence)
    def forward(p,b,image,history,body,h,dt):
        return functional_call(wrapper,(p,b),(image,history,body,h,dt))
    return vmap(forward)(params,buffers,*args,hidden,elapsed)


class TrainingPolicyBatch:
    def __init__(self, learners):
        self.learners=learners
        self.indices=[i for i,l in enumerate(learners) if not l.c.no_learn]
        self.pending=[]

    def prepare(self,states):
        buckets=defaultdict(list)
        for i in self.indices:
            l=self.learners[i];c=l.c
            key=(c.width,c.depth,c.image_size,c.recurrent_size,c.recurrent_layer,c.cnn_extra,
                 l.n_actions,len(l.context) if c.recurrent_size else 0)
            buckets[key].append(i)
        self.pending=[]; result={}
        for indices in buckets.values():
            ls=[self.learners[i] for i in indices]
            sequences=[(l.context if l.c.recurrent_size else [])+[states[i]] for i,l in zip(indices,ls)]
            out=evaluate(self.learners,indices,'actor',sequences,[l.anchors['actor'] for l in ls])
            logits=out[0][:,0];explore=logits.new_tensor([l.c.exploration for l in ls])[:,None]
            p=(1-explore)*logits.softmax(-1)+explore/logits.shape[-1]
            dist=Categorical(probs=p,validate_args=False)
            values=dist.probs.detach().cpu().numpy().astype(float)
            for i,p in zip(indices,values): result[i]=p/p.sum()
            self.pending.append((indices,sequences,out,dist))
        return result

    def learn(self,states,actions,valences,next_states,elapsed,terminals):
        metrics={}
        for indices,sequences,actor,dist in self.pending:
            ls=[self.learners[i] for i in indices]
            critic=evaluate(self.learners,indices,'critic',sequences,[l.anchors['critic'] for l in ls])
            value=critic[0][:,0,0]
            with torch.no_grad():
                # Terminal rows use an arbitrary valid input and are masked to zero.
                boot=evaluate(self.learners,indices,'critic',
                    [[states[i] if next_states[i] is None else next_states[i]] for i in indices],
                    [critic[1][j].detach() for j in range(len(indices))],sequence=False)[0][:,0,0]
                boot=boot.masked_fill(torch.tensor([terminals[i] for i in indices],device=value.device),0)
                delta=value.new_tensor([valences[i] for i in indices])+value.new_tensor([l.c.gamma**elapsed for l in ls])*boot-value.detach()
            entropy=dist.entropy()
            objective=dist.log_prob(torch.tensor([actions[i] for i in indices],device=value.device))
            objective=objective+entropy*delta.sign()*value.new_tensor([l.c.entropy for l in ls])
            parameters=[p for l in ls for u in (l.actor_update,l.critic_update) for p in u.parameters]
            gradients=torch.autograd.grad((objective.sum(),value.sum()),parameters)
            rows=torch.stack((delta,value.detach(),boot,entropy.detach(),dist.probs.detach().max(-1).values),1).detach().cpu().tolist()
            offset=0
            for j,(i,l,row) in enumerate(zip(indices,ls,rows)):
                m=dict(zip(('td','value','bootstrap','entropy','max_probability'),row))
                l.advance_memory(states[i],actor[2][j],critic[2][j])
                decay=0. if l.previous_elapsed is None else (l.c.gamma*l.trace_discount)**l.previous_elapsed
                for name,u in (('actor',l.actor_update),('critic',l.critic_update)):
                    count=len(u.parameters);g=gradients[offset:offset+count];offset+=count
                    m.update({name+'_'+k:v for k,v in u.step(g,row[0],decay).items()})
                l.previous_elapsed=elapsed
                if terminals[i]:l.clear()
                metrics[i]=m
        self.pending=[]
        return metrics
