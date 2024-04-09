
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from torch.distributions import normal

class Actor(nn.Module):

    def __init__(self, params, input_size, output_size, n_hidden, hidden_size):
        super(Actor, self).__init__()

        # absorb all parameters to self
        for key, val in params.items():  setattr(self, key, val)
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size

        if (self.n_hidden == 2):
            self.actor = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=1)
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        out = self.actor(x)
        return out    

class ExperienceReplayMemory:
    def __init__(self, capacity, config, input_state):
        self.config = config
        self.capacity = capacity
        self.input_state = input_state

        if (self.config.reputation_enabled == 0):
            self.input_act = 1
        else: 
            self.input_act = 2

        self.reset()
        
    def reset(self):
        self._states = torch.zeros((self.config.batch_size,self.config.num_game_iterations,self.input_state))
        self._actions = torch.zeros((self.config.batch_size,self.config.num_game_iterations,1))
        self._rewards = torch.zeros((self.config.batch_size,self.config.num_game_iterations,self.config.num_objectives))
        self._logprobs = torch.zeros((self.config.batch_size,self.config.num_game_iterations,1))
        self._next_states = torch.zeros((self.config.batch_size,self.config.num_game_iterations,1))
        self._dones = torch.zeros((self.config.batch_size,self.config.num_game_iterations,1), dtype=torch.bool)

    def __len__(self):
        return len(self._states)

    
class MOReinforce():

    def __init__(self, params, idx=0):
        for key, val in params.items(): setattr(self, key, val)

        if (self.reputation_enabled == 0):
            self.input_act = 1
        else: 
            self.input_act = 2

        if (self.old_actions_in_input == True):
            self.input_act += self.num_active_agents-1 # I add as input the old actions of the agents I an playing against

        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(params.device)
    
        self.optimizer = torch.optim.Adam(self.policy_act.parameters(), lr=self.lr_actor)
        self.memory = ExperienceReplayMemory(self.num_game_iterations, params, self.input_act)
        self.scheduler_act = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.n_update = 0.
        self.baseline = 0.

        self.reputation = torch.Tensor([1.])
        self.old_reputation = self.reputation
        self.previous_action = torch.Tensor([1.])

        self.is_dummy = False
        self.idx = idx

        self.eps = np.finfo(np.float32).eps.item()

        self.eps_batch = 0.00001

        self.w = torch.ones(params.num_objectives)
        if (self.betas_from_distrib):
            d = normal.Normal(1., self.sigma_beta)
            self.beta = d.sample()
            if (self.beta < 0.):
                self.beta = -self.beta          
        else:
            self.beta = self.betas[self.idx]
        print("beta=", self.beta)

    def reset(self):
        self.memory.reset()
        self.memory.i = 0
        self.accrued_reward_observation = torch.zeros(self.num_objectives)
            
    def select_action(self, state_act, _eval=False):

        self.state_act = state_act.view(-1,self.input_act)
        #print("self.state_act=", self.state_act)

        out = self.policy_act(self.state_act)
        #print("out=", out)
        dist = Categorical(out)

        if (_eval == True):
            act = torch.argmax(out).detach().unsqueeze(0)
        else:
            act = dist.sample().detach()
        logprob = dist.log_prob(act) # negativi

        self.previous_action = act
        
        return act, logprob

    def get_action_distribution(self, state):
        with torch.no_grad():
            state = state.view(-1,self.input_act)
            out = self.policy_act(state)
            return out
        
    def append_to_replay(self, i_batch, s, a, r, s_, l, d):
        self.memory._states[i_batch, self.memory.i] = s
        self.memory._actions[i_batch, self.memory.i] = a
        self.memory._rewards[i_batch, self.memory.i] = r
        self.memory._logprobs[i_batch, self.memory.i] = l
      
        self.memory.i += 1

    def beta_utility(self, rewards):
        #print("rewards=", rewards.shape)
        #print("self.w=", self.w.shape)
        utility = torch.zeros(rewards.shape[0],rewards.shape[1])
        for i in range(rewards.shape[1]):
            #print("rewards[:,i]=", rewards[:,i], rewards[:,i].shape)
            #print("rewards[:,i] @ self.w=",rewards[:,i] @ self.w, (rewards[:,i] @ self.w).shape)
            #print("utility[:,i]=",utility[:,i].shape)
            #print("utility[:,i]=",utility[:,i].shape)
            a = self.w[0]*(rewards[:,:,0]**self.beta)
            b = self.w[1]*rewards[:,:,1]
            #print("rewards[:,0]**self.beta=", a.shape)
            #print("self.w[1]*rewards[:,1]=", b.shape)
            utility[:,:] = self.w[0]*(rewards[:,:,0]**self.beta) + self.w[1]*rewards[:,:,1]
        return utility

    def compute_future_discounted_rewards(self):
        R = torch.zeros((self.batch_size, self.num_game_iterations, self.num_objectives))
        #print("R shape=", R.shape)

        R[:,self.num_game_iterations-1] = self.memory._rewards[:,self.num_game_iterations-1]
        for r in range(self.num_game_iterations-2,-1,-1):
            R[:,r] = self.memory._rewards[:,r] + R[:,r+1]*self.gamma
        #print("R=",R)
        self.baseline = torch.mean(R, dim=0)
        #print("baseline=", self.baseline)
        #print("R-baseline=", R-self.baseline)

        return R
    
    def update_mo_ser(self):
        R = self.compute_future_discounted_rewards()
        
        #print("states=", self.memory._states)
        #print("actions=", self.memory._actions)
        #print("rewards=", self.memory._rewards)
        #print("log probs=", self.memory._logprobs)
        #print("R=",R)
        #ser = self.beta_utility(self.memory._logprobs * (R-self.baseline))
        ser = self.beta_utility(self.memory._logprobs * R)
        #print("ser=", ser, ser.shape)
        loss_batch = -torch.mean(ser, dim=0)
        #print("loss_batch=", loss_batch, loss_batch.shape)
        loss = torch.mean(loss_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()