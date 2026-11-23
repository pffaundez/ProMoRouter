"""Paired query-masked episodes; only pseudo-new candidate history differs."""
import hashlib
import json
import random

import torch


class EpisodeSampler:
    def __init__(self, query_count, fitting_indices, prompts, models, seed=1, support_size=8):
        self.query_count=query_count;self.fit=tuple(fitting_indices)
        self.prompts=prompts;self.models=models;self.support_size=support_size
        if (not self.fit or len(set(self.fit))!=len(self.fit)
            or not all(0<=i<query_count for i in self.fit)
            or query_count-len(self.fit)<support_size or min(prompts,models)<1):
            raise ValueError('invalid episode population or insufficient reserved support')
        self.rng=random.Random(seed)
        self.generator=torch.Generator(device='cpu').manual_seed(seed)
        self.order=[];self.draws=0;self.digest=hashlib.sha256()

    def sample(self):
        if not self.order:
            self.order=list(range(self.models));self.rng.shuffle(self.order)
        pseudo=self.order.pop()
        target_q=torch.zeros(self.query_count,dtype=torch.bool)
        sampled=torch.rand(len(self.fit),generator=self.generator,device='cpu')>=.5
        if not bool(sampled.any()):sampled[0]=True
        target_q[list(self.fit)]=sampled
        available=[i for i in range(self.query_count) if not bool(target_q[i])]
        support=sorted(self.rng.sample(available,self.support_size))
        support_q=torch.zeros_like(target_q);support_q[support]=True
        mask=target_q[:,None,None].expand(-1,self.prompts,self.models).reshape(-1)
        control=~mask
        episodic=control.reshape(self.query_count,self.prompts,self.models).clone()
        episodic[:,:,pseudo] &= support_q[:,None]
        episodic=episodic.reshape(-1)
        if bool((mask & episodic).any()) or bool((target_q & support_q).any()):
            raise RuntimeError('target outcomes became visible in an episode')
        observed=episodic.reshape(self.query_count,self.prompts,self.models).sum((0,1))
        if int(observed[pseudo])!=self.support_size*self.prompts:
            raise RuntimeError('pseudo-new support does not have the exact action budget')
        keep=[j for j in range(self.models) if j!=pseudo]
        if not torch.equal(episodic.reshape(-1,self.models)[:,keep],control.reshape(-1,self.models)[:,keep]):
            raise RuntimeError('episode changed history of another candidate')
        self.digest.update(bytes([pseudo]));self.digest.update(mask.numpy().tobytes())
        self.digest.update(json.dumps(support).encode())
        self.draws+=1
        return dict(pseudo_index=pseudo,support_indices=support,target_queries=target_q,
                    predict=mask,control_visible=control,episodic_visible=episodic,
                    visible_per_model=observed.tolist())

    @property
    def plan_sha256(self):
        return self.digest.hexdigest()
