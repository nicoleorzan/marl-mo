
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from torch.distributions import normal
from morl_baselines.common.networks import layer_init, mlp

def _hidden_layer_init(layer):
    layer_init(layer, weight_gain=np.sqrt(2), bias_const=0.0)

def _critic_init(layer):
    layer_init(layer, weight_gain=1.0)

def _value_init(layer):
    layer_init(layer, weight_gain=0.01)

class ActorCritic(nn.Module):

    def __init__(self, params, input_size, output_size, n_hidden, hidden_size):
        super(ActorCritic, self).__init__()

        # absorb all parameters to self
        for key, val in params.items():  setattr(self, key, val)
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size

        self.actor = mlp(
            input_dim = self.input_size,
            output_dim = self.output_size,
            net_arch = [self.hidden_size, self.hidden_size],
            activation_fn = nn.Tanh,
        )
        self.actor.apply(_hidden_layer_init)
        _value_init(list(self.actor.modules())[-1])

        self.critic = mlp(
        input_dim = self.input_size,
        output_dim = self.num_objectives,
        net_arch = [self.hidden_size, self.hidden_size],
        activation_fn = nn.Tanh,
        )
        self.critic.apply(_hidden_layer_init)
        _critic_init(list(self.critic.modules())[-1])
                
    def get_value(self, x):
        out = self.critic(x)
        return out    

    def get_action_and_value(self, obs, action=None, _eval=False):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        #print("logits=", logits)
        if action is None:
            action = probs.sample()
        #print("action=", action.shape)
        #print("probs.log_prob(action)=", probs.log_prob(action))
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)
    

class ExperienceReplayMemory:
    def __init__(self, capacity, config, input_state):
        self.config = config
        self.capacity = capacity
        self.input_state = input_state

        if (self.config.reputation_enabled == 0):
            self.input = 1
        else: 
            self.input = 2

        self.reset()
        
    def reset(self):
        self._states = torch.zeros((self.config.batch_size,self.config.num_game_iterations,self.input_state))
        self._actions = torch.zeros((self.config.batch_size,self.config.num_game_iterations,1))
        self._values = torch.zeros((self.config.batch_size,self.config.num_game_iterations,self.config.num_objectives))
        self._rewards = torch.zeros((self.config.batch_size,self.config.num_game_iterations,self.config.num_objectives))
        self._logprobs = torch.zeros((self.config.batch_size,self.config.num_game_iterations,1))
        self._next_states = torch.zeros((self.config.batch_size,self.config.num_game_iterations,1))
        self._dones = torch.zeros((self.config.batch_size,self.config.num_game_iterations,1), dtype=torch.bool)

    def __len__(self):
        return len(self._states)

    
class MOActorCritic():

    def __init__(self, params, idx=0):
        for key, val in params.items(): setattr(self, key, val)

        if (self.reputation_enabled == 0):
            self.input = 1
        else: 
            self.input = 2

        if (self.old_actions_in_input == True):
            self.input += self.num_active_agents-1 # I add as input the old actions of the agents I an playing against

        self.policy = ActorCritic(params=params, input_size=self.input, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(params.device)
    
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.memory = ExperienceReplayMemory(self.num_game_iterations, params, self.input)
        self.scheduler_act = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.n_update = 0.
        self.baseline = 0.

        self.reputation = torch.Tensor([1.])
        self.old_reputation = self.reputation
        self.previous_action = torch.Tensor([1.])

        self.is_dummy = False
        self.idx = idx

        self.eps = np.finfo(np.float32).eps.item()
        self.softmax = nn.Softmax(dim=1)

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
            
    def select_action(self, state, _eval=False):

        state = state.view(-1,self.input)
        #print("self.state_act=", self.state_act)

        out = self.policy(state)
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
            state = state.view(-1,self.input)
            out = self.policy.actor(state)
            #print("out=", out)
            out = self.softmax(out)
            #print("out=", out)
            return out
        
    def append_to_replay(self, i_batch, s, a, r, s_, l, v, d):
        self.memory._states[i_batch, self.memory.i] = s
        self.memory._actions[i_batch, self.memory.i] = a
        self.memory._rewards[i_batch, self.memory.i] = r
        self.memory._logprobs[i_batch, self.memory.i] = l
        self.memory._values[i_batch, self.memory.i] = v
      
        self.memory.i += 1

    def beta_utility(self, rewards):
        #print("rewards=", rewards.shape)
        #print("self.w=", self.w.shape)
        utility = torch.zeros(rewards.shape[0],rewards.shape[1])
        for i in range(rewards.shape[1]):
            if (self.num_objectives == 1):
                utility[:,:] = rewards[:,:,0]
            elif (self.num_objectives == 2):
                utility[:,:] = self.w[0]*(rewards[:,:,0]**self.beta) + self.w[1]*rewards[:,:,1]
        return utility


    def compute_future_discounted_rewards(self):
        R = torch.zeros((self.batch_size, self.num_game_iterations, self.num_objectives))
        #print("R shape=", R.shape)

        R[:,self.num_game_iterations-1] = self.memory._rewards[:,self.num_game_iterations-1]
        for r in range(self.num_game_iterations-2,-1,-1):
            R[:,r] = self.memory._rewards[:,r] + R[:,r+1]*self.gamma
        self.baseline = torch.mean(R, dim=0)

        return R
    
    def update(self):
        #R = self.compute_future_discounted_rewards()
        
        #print("states=", self.memory._states)
        #print("actions=", self.memory._actions)
        #print("rewards=", self.memory._rewards)
        #print("log probs=", self.memory._logprobs)
        #print("R=",R)
        ser = self.beta_utility(self.memory._logprobs * (self.memory._values - self.baseline))
        #ser = self.beta_utility(self.memory._logprobs * R)
        #print("ser=", ser, ser.shape)
        loss_batch = -torch.mean(ser, dim=0)
        #print("loss_batch=", loss_batch, loss_batch.shape)
        loss = torch.mean(loss_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()