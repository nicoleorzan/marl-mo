
from src.algos.Actor import Actor
import torch
from collections import deque
from torch.distributions import Categorical
import numpy as np

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()
        
    def reset(self):
        self._states = torch.empty((self.capacity,1))
        self._actions = torch.empty((self.capacity,1))
        self._rewards = torch.empty((self.capacity,1))
        self._logprobs = torch.empty((self.capacity,1))
        self._next_states = torch.empty((self.capacity,1))
        self._dones = torch.empty((self.capacity,1), dtype=torch.bool)
        self.i = 0

    def __len__(self):
        return len(self._states)

    
class Reinforce():

    def __init__(self, params, idx=0):

        for key, val in params.items(): setattr(self, key, val)

        #self.input_act = self.obs_size
        if (self.reputation_enabled == 0):
            self.input_act = 1
        else: 
            self.input_act = 2
        print("input_act=",self.input_act)

        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(params.device)
    
        self.optimizer = torch.optim.Adam(self.policy_act.parameters(), lr=self.lr_actor)
        self.memory = ExperienceReplayMemory(self.num_game_iterations)
        self.scheduler_act = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.n_update = 0.
        self.baseline = 0.

        self.reputation = torch.Tensor([1.])
        self.old_reputation = self.reputation
        self.previous_action = torch.Tensor([1.])

        self.is_dummy = False
        self.idx = idx
        self.batch_size = self.num_game_iterations

        self.eps = np.finfo(np.float32).eps.item()

        self.eps_batch = 0.00001

    def reset(self):
        self.memory.reset()
        self.memory.i = 0
        self.return_episode = 0

    def temperature_scaled_softmax(logits, temperature=1.0):
        logits = logits / temperature
        return torch.softmax(logits, dim=0)
            
    def select_action(self, _eval=False):

        #print(" self.state_act", self.state_act)
        self.state_act = self.state_act.view(-1,self.input_act)
        #print("self.state_act=", self.state_act)

        out = self.policy_act.get_distribution(state=self.state_act)
        #print("out=", out)
        dist = Categorical(out)

        if (_eval == True):
            act = torch.argmax(out).detach()
        else:
            act = dist.sample().detach()
        logprob = dist.log_prob(act) # negativi
        #print("act=", act, "logprob=", logprob)
        
        return act, logprob

    def get_action_distribution(self, state):

        with torch.no_grad():
            out = self.policy_act.get_distribution(state)
            return out

    def append_to_replay(self, s, a, r, s_, l, d):
        self.memory._rewards[self.memory.i] = r
        self.memory._logprobs[self.memory.i] = l
        self.memory.i += 1

    def read_distrib_no_reputation(self, possible_states, n_possible_states):
        dist = torch.full((n_possible_states,2),  0.)
        for idx_s, state in enumerate(possible_states):
            dist[idx_s,:] = self.policy_act.get_distribution(state.view(-1,self.input_act))
        return dist

    def read_distrib(self, possible_states, n_possible_states):
        dist = torch.full((n_possible_states, 2),  0.)
        for idx_s, state in enumerate(possible_states):
            dist[idx_s,:] = self.policy_act.get_distribution(state.view(-1,self.input_act))
        return dist

    def update(self):
        R = 0; returns = deque()
        policy_loss = []
        
        #print("self.memory._rewards=",self.memory._rewards)
        for r in list(self.memory._rewards)[::-1]:
            R = r + R*self.gamma
            returns.appendleft(R)
        #print("returns=", returns)

        returns = torch.tensor(returns)
        baseline = torch.mean(returns)
        
        for log_prob, R in zip(self.memory._logprobs, returns):
            val = -log_prob * (R - baseline)
            policy_loss.append(val.reshape(1))

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.reset()

        return policy_loss.detach()