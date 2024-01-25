from src.algos.Actor import Actor
import copy
import torch
import random
import numpy as np
from itertools import product
from collections import namedtuple

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ExperienceReplayMemory:
    def __init__(self, params, input_state):

        for key, val in params.items(): setattr(self, key, val)
        self.capacity = self.num_game_iterations
        self.input_state = input_state
        self.reset()

    def reset(self):
        self._states = torch.empty((self.capacity,self.input_state))
        self._actions = torch.empty((self.capacity,1))
        self._rewards = torch.zeros((self.capacity,self.n_active_agents,self.num_objectives))
        self._next_states = torch.empty((self.capacity,self.input_state))
        self._dones = torch.empty((self.capacity,1), dtype=torch.bool)
        self.i = 0

    def __len__(self):
        return len(self._states)


class MoDQN():
    def __init__(self, params, idx=0):
        for key, val in params.items(): setattr(self, key, val)

        self.input_act = self.obs_size
        #if (self.reputation_enabled == 0):
        #    self.input_act = 1
        #else: 
        #    self.input_act = 2
        print("input_act=",self.input_act)
        print("agents functs=", self.agents_functions)
        print("action_size=",self.action_size)

        _output_size = self.num_objectives*self.n_active_agents*self.action_size**self.n_active_agents
        print("ouput_size=", _output_size)

        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=_output_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(device)
        self.policy_act_target = copy.deepcopy(self.policy_act).to(device)
        self.policy_act_target.load_state_dict(self.policy_act.state_dict())

        self.optimizer = torch.optim.RMSprop(self.policy_act.parameters())
        self.memory = ExperienceReplayMemory(params, self.input_act)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.reputation = torch.Tensor([1.])
        self.old_reputation = self.reputation
        self.previous_action = torch.Tensor([1.])

        self.is_dummy = False
        self.idx = idx
        self.batch_size = self.num_game_iterations

        self.action_selections = [0 for _ in range(self.action_size)]
        self.action_log_frequency = 1.

        self.update_count = 0
        if (self.decaying_epsilon == True):
            self.eps0 = self.epsilon
            self.final_epsilon = 0.001
            self.epsilon_delta = (self.eps0 - self.final_epsilon)/self.num_epochs
        self.epsilon = self.eps0
        self.r = 1.-np.exp(np.log(self.final_epsilon/self.eps0)/self.num_epochs)

        self.w = torch.Tensor(self.weights)
        self.w_agents = torch.Tensor(self.weights_agents)

        # scalarization function for the reward vector of the single agent
        if (self.scalarization_function == "linear"):
            self.scal_func = self.linear
        elif (self.scalarization_function == "ggf"):
            self.scal_func = self.GGF
        
        # scalarization functions used for reward vector of the agents he controls
        self.scal_func_agents = []
        for i in range(self.n_agents):
            if (self.agents_functions[i] == "linear"):
                self.scal_func_agents.append(self.linear)
            elif (self.agents_functions[i]  == "ggf"):
                self.scal_func_agents.append(self.GGF)

        self._print = False

        self.possibilities = list(product(range(0,2), repeat=self.n_active_agents))

    def reset(self):
        self.memory.reset()
        self.memory.i = 0
        self.idx_mf = 0

    def argmax(self, q_values):
        #if (self.idx == 0):
        #    print("q_val=", q_values)
        q_values = q_values.squeeze(0) # eliminating batch_size
        #if (self.idx == 0):
        #    print("q_val=", q_values)
        top = torch.Tensor([-10000000])
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return random.choice(ties)
        
    def select_action(self, state_act, _eval=False):
        if (self._print == True and self.idx == 0):
            print("take action")

        #self.state_act = self.state_act.view(-1,self.input_act)
        #print("bef state_act=",state_act)
        state_act = state_act.view(-1,self.input_act)
        #print("state_act=",state_act)

        if (_eval == True):
            # WE HAVE TO ADD 1 TO COMPENSATE FOR THE BATCH SIZE!!!
            act_values = self.policy_act.get_values(state=state_act)[0].view(1,  self.num_objectives, self.n_active_agents, self.action_size**self.n_active_agents) 
            scalarized_action_val_agents = torch.zeros(1, self.n_active_agents, self.action_size**self.n_active_agents)
            for i in range(self.n_active_agents):
                scalarized_action_val_agents[:,i] = self.scal_func_agents[i](act_values[:,:,i], self.w_agents)
            scalarized_action_val = self.scal_func(scalarized_action_val_agents, self.w, _dim=1)
            action = self.argmax(scalarized_action_val)
            #action = self.argmax(self.policy_act.get_values(state=state_act)[0])
        
        elif (_eval == False):
            if torch.rand(1) < self.epsilon:
                action = random.choice([i for i in range(self.action_size)])
                if (self._print == True and self.idx == 0):
                    print("====> RAND")
                    print("action=", action)
            else:
                act_values = self.policy_act.get_values(state=state_act)[0].view(1, self.num_objectives, self.n_active_agents, self.action_size**self.n_active_agents) 

                # first I scalarize the agnet's objectives
                scalarized_action_val_agents = torch.zeros(1, self.n_active_agents, self.action_size**self.n_active_agents)
                for i in range(self.n_active_agents):
                    scalarized_action_val_agents[:,i] = self.scal_func_agents[i](act_values[:,:,i], self.w_agents)

                # then I scalarize for the agents
                scalarized_action_val = self.scal_func(scalarized_action_val_agents, self.w, _dim=1)

                action = self.argmax(scalarized_action_val)
                if (self._print == True and self.idx == 0):
                    print("act_values=", act_values.shape)
                    print("scalarized_action_val_agents=", scalarized_action_val_agents.shape)
                    print("scalarized_action_val=",scalarized_action_val.shape)

        return torch.Tensor([action])
        
    def linear(self, x, w, _dim=0):
        #print("w=", w)
        #print("x=", x)
        out = torch.matmul(w, x)
        #print("out=", out)
        return out
        
    def GGF(self, x, w, _dim):
        #print("w=", w.shape)
        #print("x=", x.shape)
        #_dim=1 # WE CONSIDER DIM-0 AS THE BATCH SIZE
        #print("dim=", _dim)
        x_up = x.sort(dim=_dim)[0]
        #print("x up=", x_up.shape)
        ggf = torch.matmul(w, x_up)
        #print("ggf=", ggf)

        return ggf

    def to_actions(self, a):
        actions = torch.empty(self.n_active_agents)
        for i in range(self.n_active_agents):
            actions[i] = torch.Tensor([self.possibilities[int(a)][i]])
        return actions

    def append_to_replay(self, s, a, r, s_, d, act_ag_idx):
        self.memory._states[self.memory.i] = s
        self.memory._actions[self.memory.i] = a
        self.memory._next_states[self.memory.i] = s_
        self.memory._dones[self.memory.i] = d

        for a in range(self.n_active_agents):
            self.memory._rewards[self.memory.i, a] = r['agent_'+str(act_ag_idx[a])]  # like tensor([9.8826, 0.0000])

        #print("self.memory._rewards",self.memory._rewards)
        self.memory.i += 1

    def prep_minibatch(self):
        batch_state = self.memory._states
        batch_action = self.memory._actions.long()
        batch_reward = self.memory._rewards
        batch_next_state = self.memory._next_states
        batch_dones = self.memory._dones
        #print("batch_next_state=",batch_next_state)
        #print("batch_dones=",batch_dones)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(-1,1)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        #print('non_final_next_states=',non_final_next_states)
        #print("non_final_mask", non_final_mask)
        #print("empty_next_state_values=",empty_next_state_values)
        return batch_state, batch_action, batch_reward, batch_dones, non_final_next_states, non_final_mask, empty_next_state_values

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, batch_dones, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars
        #print("batch_state=",batch_state)
        
        current_q_values = self.policy_act.get_values(batch_state).view(self.batch_size, self.num_objectives, self.n_active_agents, self.action_size**self.n_active_agents) 
        if (self._print == True and self.idx == 0):
            print("current_q_values",current_q_values.shape)
            #print("scal val used to take act", self.scal_func(current_q_values, self.w))
        
        #print("batch_action=",batch_action)
        batch_action = batch_action.repeat(1, self.num_objectives*self.n_active_agents).view(self.batch_size, self.num_objectives, self.n_active_agents, 1)
        current_q_values = torch.gather(current_q_values, dim=3, index=batch_action)
        if (self._print == True and self.idx == 0):
            print("batch_action.repeat=",batch_action.shape)
            print("vals filtered by acts=",current_q_values.shape)
            
        #compute target
        with torch.no_grad():
            max_next_q_values = torch.zeros((self.batch_size, self.num_objectives, self.n_active_agents, 1), device=self.device, dtype=torch.float)#.unsqueeze(dim=1)
            #print("max_next_q_values (empty)=",max_next_q_values)
            if not empty_next_state_values:
                # first we gotta compute MAX NEXT ACTION
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                if (self._print == True and self.idx == 0):
                    print("non_final_next_states=",non_final_next_states.shape)
                    print("max_next_action=",max_next_action, max_next_action.shape)
                max_next_action = max_next_action.repeat(1, self.num_objectives).view(self.batch_size, self.num_objectives, 1)
                dist = self.policy_act_target.get_values(state=non_final_next_states).view(self.batch_size, self.num_objectives, self.n_active_agents, 1)
                if (self._print == True and self.idx == 0):
                    print("max_next_action reshape=",max_next_action, max_next_action.shape)
                    print("dist=",dist, dist.shape)
                #print("dist.gather(1, max_next_action)=",dist.gather(1, max_next_action),dist.gather(1, max_next_action).shape )
                max_next_q_values[non_final_mask] = dist.gather(2, max_next_action)
                for i in range(len(batch_dones)):
                    if (batch_dones[i] == torch.Tensor([True])):
                        max_next_q_values[i] = 0.
                ##print("current_q_values=",current_q_values, current_q_values.shape)
            if (self._print == True and self.idx == 0):
                print("max_next_q_values=", max_next_q_values.shape)
                print("self.gamma*max_next_q_values=",(self.gamma*max_next_q_values).shape)

            #scalarize batch rewards
            batch_reward_for_update = torch.zeros((self.batch_size, self.num_objectives, self.n_active_agents, 1))
            batch_scal_rewards_agents = torch.zeros((self.batch_size, self.n_active_agents))
            for j in range(self.batch_size):
                for i in range(self.n_active_agents):
                    #print("self.scal_func_agents[i](batch_reward[:,i,:])=",self.scal_func_agents[i](batch_reward[j,i,:], self.w_agents))
                    batch_scal_rewards_agents[j,i] = self.scal_func_agents[i](batch_reward[j,i,:],self.w_agents)
                #print('batch_scal_rewards[:,i] =',batch_scal_rewards[:,i] )
            #print("batch_scal_rewards_agents=", batch_scal_rewards_agents.shape)

            batch_scal_rewards = torch.zeros((self.batch_size))
            for j in range(self.batch_size):
                #print("batch_scal_rewards_agents[i]=",batch_scal_rewards_agents[j])
                batch_scal_rewards[j] = self.scal_func(batch_scal_rewards_agents[j], self.w, _dim=0)
                #print('batch_scal_rewards=',batch_scal_rewards)
                batch_reward_for_update[j, :, :] = batch_scal_rewards[j]
                #print('batch_reward_for_update=',batch_reward_for_update)
            #print("batch_scal_rewards=",batch_scal_rewards)

            # probably, I have to repeat the final reward for all the state-action stuff I have. So I use repeat
            #batch_scal_rewards = batch_scal_rewards.repeat(self.num_objectives*self.n_active_agents)#.unsqueeze(-1)
            #print("batch_scal_rewards=",batch_scal_rewards.shape)
            
            expected_q_values = batch_scal_rewards + self.gamma*max_next_q_values
            if (self._print == True and self.idx == 0):
                print("batch_reward_for_update.shape=", batch_reward_for_update.shape)
                print("batch_reward=",batch_reward.shape)
                print("expected_q_values=",expected_q_values.shape)

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, _iter):

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_model()

        self.reset()
        self.scheduler.step()
        #print("LR=",self.scheduler.get_last_lr())

        #modif exploration
        self.update_epsilon(_iter)

        return loss.detach()
    
    def update_epsilon(self, _iter):
        #print("update epsilon")
        #self.epsilon = self.eps0*(1.-self.r)**_iter
        if (self.epsilon >= self.final_epsilon): 
            self.epsilon = self.epsilon -self.epsilon_delta #self.eps0*(1.-self.r)**_iter

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            print("updating target model")
            self.policy_act_target.load_state_dict(self.policy_act.state_dict())

    def get_max_next_state_action(self, next_states):
       
        next_vals = self.policy_act_target.get_values(state=next_states).view(self.batch_size, self.num_objectives, self.n_active_agents, self.action_size**self.n_active_agents) 
        scalarized_next_vals = self.scal_func(next_vals, self.w)
        max_vals = scalarized_next_vals.max(dim=1)[1]#.view(-1, 1)
        if (self._print == True and self.idx == 0):
            print("next_vals=", next_vals)
            print("scalarized_next_vals=", scalarized_next_vals)
            print("max_vals=",max_vals)
        self._print == True 
        return max_vals
    
    def MSE(self, x):
        return 0.5 * x.pow(2)