from algos.Actor import Actor
import copy
import torch
import random
import numpy as np
from torch.distributions import normal
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
        self._rewards = torch.empty((self.capacity,self.num_objectives))
        self._next_states = torch.empty((self.capacity,self.input_state))
        self._dones = torch.empty((self.capacity,1), dtype=torch.bool)
        self.i = 0

    def __len__(self):
        return len(self._states)


class MoDQN():
    def __init__(self, params, idx=0):
        for key, val in params.items(): setattr(self, key, val)

        #self.input_act = self.obs_size
        if (self.reputation_enabled == 0):
            self.input_act = 1
        else: 
            self.input_act = 2
        
        if (self.old_actions_in_input == True):
            self.input_act += self.num_active_agents-1 # I add as input the old actions of the agents I an playing against
        print("input_act=",self.input_act)

        _output_size = self.action_size*self.num_objectives

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
        if (self.scalarization_function == "linear"):
            self.scal_func = self.linear
        elif (self.scalarization_function == "ggf"):
            self.scal_func = self.GGF
        elif (self.scalarization_function == "non-linear-pgg"):
            self.scal_func = self.non_linear_pgg
        elif (self.scalarization_function == "sigmoid"):
            self.scal_func = self.sigmoid

        self._print = False

        if (self.betas_from_distrib):
            d = normal.Normal(1., self.sigma_beta)
            self.beta = d.sample()
            #if (self.beta < 0.):
            #    self.beta = -self.beta          
        else:
            self.beta = self.betas[self.idx]
        #print("beta=", self.beta)

        self.reset()

    def reset(self):
        self.memory.reset()
        self.memory.i = 0
        self.previous_action = torch.Tensor([0])
        self.return_episode = 0.

    def argmax(self, q_values):
        q_values = q_values.squeeze(0) # eliminating batch_size
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
        #print("state_act", state_act)
        state_act = state_act.view(-1,self.input_act)

        if (_eval == True):
            # WE HAVE TO ADD 1 TO COMPENSATE FOR THE BATCH SIZE!!!
            act_values = self.policy_act.get_values(state=state_act)[0].view(1, self.num_objectives, self.action_size) 
            scalarized_action_val = self.scal_func(act_values, self.w)
            action = self.argmax(scalarized_action_val)
        
        elif (_eval == False):
            if torch.rand(1) < self.epsilon:
                action = random.choice([i for i in range(self.action_size)])
                if (self._print == True and self.idx == 0):
                    print("take action (rand): action=", action)
            else:
                act_values = self.policy_act.get_values(state=state_act)[0].view(1, self.num_objectives, self.action_size)
                scalarized_action_val = self.scal_func(act_values, self.w)
                action = self.argmax(scalarized_action_val)
                if (self._print == True and self.idx == 0):
                    print("take action: action=", action)

        self.previous_action = action
                
        return torch.Tensor([action])
        
    def get_action_values(self, state):
        with torch.no_grad():
            out = self.policy_act.get_values(state=state).view(self.num_objectives, self.action_size) 
            return out
        
    def linear(self, x, w):
        out = torch.matmul(w, x)
        return out
    
    def non_linear_pgg(self, x, w):
        if (self.num_objectives == 3):
            out = w[0]*(x[:,0])**self.beta + w[1]*x[:,1] + w[2]*x[:,2]
        elif (self.num_objectives == 2):
            out = w[0]*(x[:,0])**self.beta + w[1]*x[:,1]
        elif (self.num_objectives == 1):
            out = w[0]*x[:,0]
        return out
    
    def sigmoid(self, x, w):
        out = (w[0]*x[:,0]*self.num_active_agents)**self.beta/self.num_active_agents + w[1]*x[:,1] + w[2]*x[:,2]
        return out
        
    def GGF(self, x, w):
        _dim=1 # WE CONSIDER DIM-0 AS THE BATCH SIZE
        x_up = x.sort(dim=_dim)[0]
        ggf = torch.matmul(w, x_up)
        return ggf

    def append_to_replay(self, s, a, r, s_, d):
        self.memory._states[self.memory.i] = s
        self.memory._actions[self.memory.i] = a
        self.memory._rewards[self.memory.i] = r  # like tensor([9.8826, 0.0000])
        self.memory._next_states[self.memory.i] = s_
        self.memory._dones[self.memory.i] = d
        self.memory.i += 1

    def prep_minibatch(self):
        batch_state = self.memory._states
        batch_action = self.memory._actions.long()
        batch_reward = self.memory._rewards
        batch_next_state = self.memory._next_states
        batch_dones = self.memory._dones
       
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
        if (self._print == True and self.idx == 0):
            print("compute loss")
        batch_state, batch_action, batch_reward, batch_dones, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars
        
        current_q_values = self.policy_act.get_values(batch_state).view(self.batch_size, self.num_objectives, self.action_size)
        if (self._print == True and self.idx == 0):
            print("batch_state=",batch_state, batch_state.shape)
            print("current_q_values", current_q_values, current_q_values.shape)
            print("scal val used to take act", self.scal_func(current_q_values, self.w), self.scal_func(current_q_values, self.w).shape)

        batch_action = batch_action.repeat(1, self.num_objectives).view(self.batch_size, self.num_objectives, 1)
        current_q_values = torch.gather(current_q_values, dim=2, index=batch_action)
        if (self._print == True and self.idx == 0):
            print("batch_actions=", batch_action, batch_action.shape)
            print("current q values filtered by acts=",current_q_values, current_q_values.shape)
            print("\n")
        if (self._print == True and self.idx == 0):   
            print("batch_reward.shape=",batch_reward, batch_reward.shape)
            print("batch rewrd with right shape: batch_reward.unsqueeze(-1)=",batch_reward.unsqueeze(-1), batch_reward.unsqueeze(-1).shape)
            print("\n")
        #compute target
        with torch.no_grad():
            max_next_q_values = torch.zeros((self.batch_size, self.num_objectives, 1), device=self.device, dtype=torch.float)#.unsqueeze(dim=1)
            #print("max_next_q_values (empty)=",max_next_q_values)
            if (self._print == True and self.idx == 0):
                print("empty_next_state_values=",empty_next_state_values)
            if not empty_next_state_values:
                # first we gotta compute MAX NEXT ACTION
                if (self._print == True and self.idx == 0):
                    print("non_final_next_states=",non_final_next_states, non_final_next_states.shape)
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                if (self._print == True and self.idx == 0):
                    print("max_next_action=",max_next_action, max_next_action.shape)
                max_next_action = max_next_action.repeat(1, self.num_objectives).view(self.batch_size, self.num_objectives, 1)
                dist = self.policy_act_target.get_values(state=non_final_next_states).view(self.batch_size, self.num_objectives, self.action_size) #.act(state=non_final_next_states, greedy=False, get_distrib=True)
                if (self._print == True and self.idx == 0):
                    print("max_next_action reshape=", max_next_action.shape)
                    print("dist=",dist, dist.shape)
                #print("dist.gather(1, max_next_action)=",dist.gather(1, max_next_action),dist.gather(1, max_next_action).shape )
                max_next_q_values[non_final_mask] = dist.gather(2, max_next_action)
                for i in range(len(batch_dones)):
                    if (batch_dones[i] == torch.Tensor([True])):
                        max_next_q_values[i] = 0.
            if (self._print == True and self.idx == 0):
                # fin qui ok
                print("max_next_q_values=",max_next_q_values, max_next_q_values.shape)
            if (self._print == True and self.idx == 0):
                print("self.gamma*max_next_q_values=",(self.gamma*max_next_q_values).shape)
                print("batch_reward.shape=",batch_reward, batch_reward.shape)

            expected_q_values = batch_reward.unsqueeze(-1) + self.gamma*max_next_q_values
            if (self._print == True and self.idx == 0):
                print("expected_q_values=",expected_q_values.shape)

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        if (self._print == True and self.idx == 0):
            print("diff=", diff.shape)
            print("mse loss=", loss.shape)
            print("loss=", loss.shape)

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

        self.update_epsilon(_iter)

        return loss.detach()
    
    def update_epsilon(self, _iter):
        if (self.epsilon >= self.final_epsilon): 
            self.epsilon = self.epsilon -self.epsilon_delta

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.policy_act_target.load_state_dict(self.policy_act.state_dict())

            if (self._print == True and self.idx == 0):
                print("updating target model")
            

    def get_max_next_state_action(self, next_states):
        next_vals = self.policy_act_target.get_values(state=next_states).view(self.batch_size, self.num_objectives,self.action_size) ##.act(state=next_states, greedy=False, get_distrib=True)
        scalarized_next_vals = self.scal_func(next_vals, self.w)
        max_vals = scalarized_next_vals.max(dim=1)[1]#.view(-1, 1)
        if (self._print == True and self.idx == 0):
            print("GET MAX NEXT STATE ACTION")
            print("next_vals=", next_vals, next_vals.shape)
            print("scalarized_next_vals=", scalarized_next_vals, scalarized_next_vals.shape)
            print("max_vals=",max_vals, max_vals.shape)
        self._print == True 
        return max_vals
    
    def MSE(self, x):
        return 0.5 * x.pow(2)