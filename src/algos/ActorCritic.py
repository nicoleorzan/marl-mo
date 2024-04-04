import torch.nn as nn
import torch
from torch.distributions import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):

    def __init__(self, params, input_size, output_size, hidden_size):
        super(ActorCritic, self).__init__()

        for key, val in params.items():  setattr(self, key, val)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.input_size, self.hidden_size)),
            #nn.Tanh(),
            #layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.input_size, self.hidden_size)),
            #nn.Tanh(),
            #layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_size, self.output_size), std=0.01),
        )
        
    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        logits = self.actor(state)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)
    
    