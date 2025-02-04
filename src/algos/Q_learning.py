
import torch
import random

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def __len__(self):
        return len(self.memory)

class Q_learning_agent():

    def __init__(self, params, idx=0):

        for key, val in params.items(): setattr(self, key, val)

        self.reputation = torch.Tensor([1.0])
        self.old_reputation = self.reputation

        self.is_dummy = False
        self.idx = idx
        print("\nAgent", self.idx)

        # Action Policy
        self.max_value = max(self.mult_fact)/(1.-self.gamma)

        if (self.reputation_enabled == 0): 
            if (len(self.mult_fact) == 1):
                input_Q = (self.action_size,)
            else: 
                input_Q = (len(self.mult_fact), self.action_size)
        else:
            if (len(self.mult_fact) == 1):
                input_Q = (2, self.action_size) # 2 is for the binary reputation
            else: 
                input_Q = (2, len(self.mult_fact), self.action_size)  # 2 is for the binary reputation
        self.Q = torch.full(input_Q, self.max_value, dtype=float)
        print("self.Q=", self.Q)

        if (self.reputation_assignment == True):
            input_Q2 = (2, 2, 2)  # get as input old reputation and action, assign new reputation
            self.Q_reputation_assignment = torch.full(input_Q2, self.max_value, dtype=float)
        
        self.memory = ExperienceReplayMemory(self.num_game_iterations)
        self.memory_rep = ExperienceReplayMemory(self.num_game_iterations)

        self.reset()

    def append_to_replay(self, s, a, r, s_, d):
        self.memory.push((s, a, r, s_, d))

    def append_to_replay_rep(self, s, a, r, s_, d):
        self.memory_rep.push((s, a, r, s_, d))

    def argmax(self, q_values):
        top = torch.Tensor([-10000000])
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return random.choice(ties)

    def reset(self):
        self.previous_action = torch.Tensor([1.])
        self.return_episode = 0

    def select_current_q(self, state_act):
        #print("self.state_act=",self.state_act)
        #print("Q=", self.Q)
        if (self.reputation_enabled == 0): 
            if (len(self.mult_fact) == 1):
                current_q = self.Q[:]
            else: 
                current_q = self.Q[state_act[0].long(),:] # Only m factor
        else:
            if (len(self.mult_fact) == 1):
                current_q = self.Q[state_act[0].long(), :] # only reputation
            else: 
                current_q = self.Q[state_act[0].long(), state_act[1].long(), :] 
        #print("current_q=", current_q)
        return current_q
    
    def select_action(self, state_act, _eval=False):
        
        current_q = self.select_current_q(state_act)
        #print("current_q=",current_q, current_q.shape)
        assert(current_q.shape == torch.Size([self.action_size]))

        if (_eval == True):
            action = self.argmax(current_q)
        elif (_eval == False):  
            if torch.rand(1) < self.epsilon:
                #print("RANDOM")
                action = random.choice([i for i in range(self.action_size)])
            else:
                #print("argmax")
                action = self.argmax(current_q)
        #print("action=", action)
        
        return torch.Tensor([action])
    
    def select_reputation_assignment(self, rep_assignment_state, _eval=False):
        
        #print("Q_reputation_assignment=",self.Q_reputation_assignment)
        current_q = self.Q_reputation_assignment[rep_assignment_state[0], rep_assignment_state[1], :]
        #print("current_q=",current_q, current_q.shape)
        assert(current_q.shape == torch.Size([2]))

        if (_eval == True):
            action = self.argmax(current_q)
        elif (_eval == False):  
            if torch.rand(1) < self.epsilon:
                #print("RANDOM")
                action = random.choice([i for i in range(self.action_size)])
            else:
                #print("argmax")
                action = self.argmax(current_q)
        #print("action=", action)
        
        return torch.Tensor([action])
    
    def get_action_distribution(self, state_act):
        return self.Q[state_act.long(),:]
    
    def update(self):
        #print("memory=", self.memory.memory)
        
        for i in range(self.num_game_iterations):
            state, action, reward, next_state, done = self.memory.memory[i]
            state = state.long()
            action = action.long()
            next_state = next_state.long()
            #print("state=", state)
            #print("action=", action)
            #print("Q=", self.Q)
            #print("self.Q[state[0], state[1], action]=", self.Q[state[0], state[1], action])
            #print("next_state=", next_state)

            if (self.reputation_enabled == 0): 
                if (len(self.mult_fact) == 1):
                    if (done):
                        self.Q[action] += self.lr_actor*(reward - self.Q[action])
                    else:
                        self.Q[action] += self.lr_actor*(reward + self.gamma*torch.max(self.Q[:]) - self.Q[action])
                else:
                    if (done):
                        self.Q[state, action] += self.lr_actor*(reward - self.Q[state, action])
                    else:
                        self.Q[state, action] += self.lr_actor*(reward + self.gamma*torch.max(self.Q[next_state, :]) - self.Q[state, action])
            else:
                if (len(self.mult_fact) == 1):
                    if (done):
                        self.Q[state, action] += self.lr_actor*(reward - self.Q[state, action])
                    else:
                        self.Q[state, action] += self.lr_actor*(reward + self.gamma*torch.max(self.Q[next_state, :]) - self.Q[state, action])
                else:
                    if (done):
                        self.Q[state[0], state[1], action] += self.lr_actor*(reward - self.Q[state[0], state[1], action])
                    else:
                        self.Q[state[0], state[1], action] += self.lr_actor*(reward + self.gamma*torch.max(self.Q[next_state[0], next_state[1], :]) - self.Q[state[0], state[1], action])

            if (self.reputation_enabled == 1 and i < len(self.memory_rep.memory)):
                state_r, action_r, reward_r, next_state_r, done_r = self.memory_rep.memory[i]
                state_r = state_r.long()
                action_r = action_r.long()
                next_state_r = next_state_r.long()
                if (done_r):
                    self.Q_reputation_assignment[state_r[0], state_r[1], action_r] += self.lr_actor*(reward_r - self.Q_reputation_assignment[state_r[0], state_r[1], action_r])
                else:
                    self.Q_reputation_assignment[state_r[0], state_r[1], action_r] += self.lr_actor*(reward_r + self.gamma*torch.max(self.Q_reputation_assignment[next_state_r[0], next_state_r[1], :]) - self.Q_reputation_assignment[state_r[0], state_r[1], action_r])

        self.memory.memory = []            
        self.memory_rep.memory = []
        self.reset()