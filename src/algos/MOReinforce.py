
from src.algos.Actor import Actor
import torch
from collections import deque
from torch.distributions import Categorical
import numpy as np

class ExperienceReplayMemory:
    def __init__(self, capacity, config):
        self.config = config
        self.capacity = capacity
        self.reset()
        
    def reset(self):
        self._states = torch.empty((self.config.num_game_iterations,1))
        self._actions = torch.empty((self.config.num_game_iterations,1))
        self._rewards = torch.empty((self.config.num_game_iterations,self.config.num_objectives))
        self._acc_rewards = torch.empty((self.config.num_game_iterations,self.config.num_objectives))
        self._logprobs = torch.empty((self.config.num_game_iterations,1))
        self._next_states = torch.empty((self.config.num_game_iterations,1))
        self._dones = torch.empty((self.config.num_game_iterations,1), dtype=torch.bool)
        self.i = 0

    def __len__(self):
        return len(self._states)

    
class MOReinforce():

    def __init__(self, params, idx=0):

        for key, val in params.items(): setattr(self, key, val)

        #self.input_act = self.obs_size
        if (self.reputation_enabled == 0):
            self.input_act = 1
        else: 
            self.input_act = 2

        if (self.num_objectives > 1): # I have to condition the state on the accrued rewards
            self.input_act = self.input_act + self.num_objectives
        print("input_act=", self.input_act)

        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(params.device)
    
        self.optimizer = torch.optim.Adam(self.policy_act.parameters(), lr=self.lr_actor)
        self.memory = ExperienceReplayMemory(self.num_game_iterations, params)
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

        if (self.utility == "linear"):
            self.w = torch.ones(params.num_objectives)

    def reset(self):
        self.memory.reset()
        self.memory.i = 0
        self.accrued_reward_observation = torch.zeros(self.num_objectives)
            
    def select_action(self, _eval=False):

        #print(" self.state_act", self.state_act)
        if (self.num_objectives > 1):
            self.state_act = torch.cat((self.state_act, self.accrued_reward_observation),0)
            #print(" self.state_act cat", self.state_act)
        self.state_act = self.state_act.view(-1,self.input_act)
        #print(" self.state_act", self.state_act)

        #print("self.state_act=", self.state_act)

        out = self.policy_act.get_distribution(state=self.state_act)
        #print("out=", out)
        dist = Categorical(out)

        if (_eval == True):
            act = torch.argmax(out).detach().unsqueeze(0)
            #print("eval act=", act)
        else:
            act = dist.sample().detach()
            #print("no eval act=", act)
        logprob = dist.log_prob(act) # negativi
        #print("act=", act, "logprob=", logprob)
        
        return act, logprob

    def get_action_distribution(self, state):

        if (self.num_objectives > 1):
            state = torch.cat(state, self.accrued_reward_observation)
        with torch.no_grad():
            out = self.policy_act.get_distribution(state)
            return out

    def append_to_replay(self, s, a, r, s_, l, d):
        self.memory._rewards[self.memory.i] = r
        self.memory._logprobs[self.memory.i] = l
        self.accrued_reward_observation += r*self.gamma**self.memory.i
        #if (self.idx == 1):
        #    print("rewards=", r)
        #    print("self.accrued_reward=",self.accrued_reward_observation )
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
    
    def linear_utility(self, rewards):
        utility = 0
        for ri, wi in zip(rewards,self.w):
            utility += ri*wi
        return utility
    
    def utility_mul(self, rewards):
        u = rewards[:,0]*rewards[:,1]
        return u.view(-1,1)
        
    def compute_accrued_rewards(self):
        accrued_rewards = torch.zeros(self.num_game_iterations+1, self.num_objectives)

        r_vals = 0
        for i, r_i in enumerate(self.memory._rewards):
            r_vals += r_i*self.gamma**i
            accrued_rewards[i+1] = r_vals

        return accrued_rewards
    
    def compute_future_discounted_rewards(self):
        R = 0; future_discounted_rewards = deque()

        for r in list(self.memory._rewards)[::-1]:
            R = r + R*self.gamma
            future_discounted_rewards.appendleft(R)

        future_discounted_rewards = torch.stack([i for i in future_discounted_rewards])

        return future_discounted_rewards
    
    def update_mo(self):
        #acc_rewards = self.compute_accrued_rewards()
        fut_rewards = self.compute_future_discounted_rewards()
        policy_loss = []
        
        for i, log_prob in enumerate(self.memory._logprobs):
            #print("acc_rewards[i]=", acc_rewards[i])
            #print("fut_rewards[i]=", fut_rewards[i])
            #print("acc_rewards[i] + fut_rewards[i]*self.gamma**i=",acc_rewards[i] + fut_rewards[i]*self.gamma**i)
            if (self.utility == "linear"):
                #val = -log_prob * self.linear_utility(acc_rewards[i] + fut_rewards[i]*self.gamma**i)
                #print("acc_rewards[i] + fut_rewards[i]*self.gamma**i=",acc_rewards[i] + fut_rewards[i]*self.gamma**i)
                #if (self.idx == 1):
                #    print("rewards=", self.memory._rewards)
                #    print("fut rewards last", fut_rewards[0])
                #    print("self.accrued_reward_observation=", self.accrued_reward_observation)
                #     print("utility=",self.linear_utility(self.accrued_reward_observation))
                val = -log_prob * self.linear_utility(self.accrued_reward_observation)
            if (self.utility == "prod"):
                #val = -log_prob * self.utility_mul(acc_rewards[i] + fut_rewards[i]*self.gamma**i)
                val = -log_prob * self.utility_mul(self.accrued_reward_observation)
            #print("val=", val)
            policy_loss.append(val.reshape(1))

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.reset()

        return policy_loss.detach()
    
    def update(self):
        R = 0; returns = deque()
        policy_loss = []
        
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
        #print("loss=", policy_loss)
        policy_loss.backward()
        self.optimizer.step()

        self.reset()

        return policy_loss.detach()
    
    def update_reward(self):

        batch_reward = self.memory._rewards
        #print("batch_rew=", batch_reward)

        policy_loss = []

        baseline = torch.mean(self.memory._rewards)
        for log_prob, rew in zip(self.memory._logprobs, batch_reward):
            #print("-logprob=", -log_prob, ", rew=", rew)
            val = -log_prob * (rew - baseline)
            policy_loss.append(val.reshape(1))

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.reset()

        return policy_loss.detach()