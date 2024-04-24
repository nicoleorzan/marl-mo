
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
            self.critic = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.num_objectives)
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=1)
            )
            self.critic = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.num_objectives)
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
        self._critic = torch.zeros((self.config.batch_size,self.config.num_game_iterations,self.config.num_objectives))
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

        self.policy = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(params.device)
    
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_actor)
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

        self.eps = 0.0001
        self.v_coef = 0.5
        self.e_coef=0.01

    def reset(self):
        self.memory.reset()
        self.memory.i = 0
        self.accrued_reward_observation = torch.zeros(self.num_objectives)
            
    def select_action(self, state_act, _eval=False):
        #print("state_act=", state_act)
        self.state_act = state_act.view(-1,self.input_act)
        #print("self.state_act=", self.state_act)

        out = self.policy.actor(self.state_act)
        #print("out=", out)
        dist = Categorical(out)
        #print("dist=", dist)

        if (_eval == True):
            act = torch.argmax(out).detach().unsqueeze(0)
        else:
            act = dist.sample().detach()
        logprob = dist.log_prob(act) # negativi

        self.previous_action = act
        #print("act=", act)

        crit = self.policy.critic(self.state_act)
        #print("crit=", crit)
        
        return act, logprob, crit

    def get_action_distribution(self, state):
        with torch.no_grad():
            state = state.view(-1,self.input_act)
            out = self.policy.actor(state)
            return out
        
    def append_to_replay(self, i_batch, s, a, r, s_, l, c, d):
        self.memory._states[i_batch, self.memory.i] = s
        self.memory._actions[i_batch, self.memory.i] = a
        self.memory._rewards[i_batch, self.memory.i] = r
        self.memory._logprobs[i_batch, self.memory.i] = l
        self.memory._critic[i_batch, self.memory.i] = c
        self.memory._dones[i_batch, self.memory.i] = d
      
        self.memory.i += 1

    def beta_utility(self, rewards):
        #print("rewards=",rewards, rewards.shape)
        #assert(rewards >= 0.) # beta utility can only work with positive numbers
        if (self.num_objectives == 2):
            collective_utility = self.w[0]*(torch.select(rewards,dim=-1,index=0)**self.beta) # self.w[0]*(rewards[0]**self.beta)
            individual_utility = self.w[1]*torch.select(rewards,dim=-1,index=1)
            utility = collective_utility + individual_utility
            #print("collective utility=",collective_utility, collective_utility.shape)
            #print("individual_utility=",individual_utility)
        if (self.num_objectives == 3):
            collective_utility = self.w[0]*(torch.select(rewards,dim=-1,index=0)**self.beta) # self.w[0]*(rewards[0]**self.beta)
            individual_utility = self.w[1]*torch.select(rewards,dim=-1,index=1)
            reputation_utility = self.w[2]*torch.select(rewards,dim=-1,index=2)
            utility = collective_utility + individual_utility + reputation_utility
        return utility

    def compute_future_discounted_rewards(self):
        R = torch.zeros((self.batch_size, self.num_game_iterations, self.num_objectives))
        standardized_rewards = (self.memory._rewards - self.memory._rewards.mean()) / (self.memory._rewards.std() + self.eps)
        
        #rewards = self.memory._rewards
        rewards = standardized_rewards

        R[:,self.num_game_iterations-1] = rewards[:,self.num_game_iterations-1]
        for r in range(self.num_game_iterations-2,-1,-1):
            # CERTI R SONO NEGATIVI PERCHE` USO GLI STANDARDISED REWARDS
            R[:,r] = rewards[:,r] + R[:,r+1]*self.gamma
        self.baseline = torch.mean(R, dim=0)

        return R
    
    def MSE(self, x):
        return 0.5 * x.pow(2)
    
    def update_mo_ser(self):
        R = self.compute_future_discounted_rewards() #R -> [batch, n_iter, n_obj]

        # ACTOR UPDATE
        #print("R=", R.shape)
        #print("self.memory._logprobs=", self.memory._logprobs.shape)
        out_critic = self.policy.critic(self.memory._states)
        #print("out_critic=", out_critic.shape) # dim = [batch, n_iter, n_obj]
        out_critic = torch.sum(out_critic, dim=1) # dim = [batch, n_obj]
        #print("out_critic (sum over iterations)=", out_critic.shape)
        u_V = self.beta_utility(out_critic)#.unsqueeze(1)
        #print("u_V=", u_V.shape)
        # ===> EXPECTATION OVER BATCH IS THE LAST THING TO DO, BC WE ARE ESTIMATING J, WE DO NOT HAVE THE REAL FUNCTION!!!!!!!
        actor_loss = -torch.mean(u_V)#.backward()
        #print("actor_loss=", actor_loss)

        #J_loss = (self.memory._logprobs*(R - self.baseline)).reshape((self.batch_size*self.num_game_iterations, self.num_objectives))
        #print("J_loss=",J_loss,J_loss.shape)

        # CRITIC UPDATE

        # update the critic!!
        # version of temporal difference:

        #print("dones=", self.memory._dones.shape)
        #print("self.memory._rewards=",self.memory._rewards.shape)
        #print("self.memory._critic=",self.memory._critic.shape)
        """
        expected_vs = torch.zeros((self.batch_size, self.num_game_iterations, self.num_objectives))
        dones = self.memory._dones
        for b in range(self.batch_size):
            for i in range(self.num_game_iterations):
                if dones[b,i] == False:
                    expected_vs[b,i,:] = self.memory._rewards[b,i,:] + self.gamma*self.memory._critic[b,i,:]
                else:
                    expected_vs[b,i,:] = self.memory._rewards[b,i,:]

        critic_TD = (expected_vs - self.memory._critic).reshape((self.batch_size*self.num_game_iterations, self.num_objectives))
        critic_loss = self.MSE(critic_TD).mean(dim=0).sum(0)"""

        advantage = R-self.memory._critic
        # get nllloss from actor
        #actor_out = self.policy.actor(self.memory._states)
        #log_prob = self.policy.log_prob(batch.action, actor_out)
        # don't propagate gradients from advantage
        actor_loss = -self.memory._logprobs*advantage.detach()
        #print("actor_loss=",actor_loss)
        #print("self.memory._critic=",self.memory._critic)
        # entropy
        #entropy = -torch.exp(self.memory._logprobs) * self.memory._logprobs
        # mseloss for critic
        critic_loss = advantage.pow(2)
        # average over objectives
        critic_loss = critic_loss.mean(-1, keepdims=True)

        # SER policy gradient update
        actor_loss = (actor_loss*self.memory._critic).sum(-1, keepdims=True)

        loss = actor_loss + self.v_coef*critic_loss# - self.e_coef*entropy
        loss = loss.reshape((self.batch_size*self.num_game_iterations)).mean()
        #print("loss=", loss, loss.shape)
        
        #print("critic_loss=", critic_loss)

        #final_loss = J_loss + actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()