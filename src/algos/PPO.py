import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.algos.ActorCritic import ActorCritic


class ExperienceReplayMemory:
    def __init__(self, params, input_state):

        for key, val in params.items(): setattr(self, key, val)
        self.capacity = self.num_game_iterations
        self.input_state = input_state

    def reset(self):
        self._states = torch.zeros((self.capacity,self.input_state))
        self._actions = torch.zeros((self.capacity,1))
        self._logprobs = torch.zeros((self.capacity,1))
        self._rewards = torch.zeros((self.capacity,1))
        self._advantages = torch.zeros((self.capacity,1))
        self._dones = torch.zeros((self.capacity,1))
        self._values = torch.zeros((self.capacity,1))
        self.i = 0

    def __len__(self):
        return len(self._states)
    

class PPO():

    def __init__(self, params, idx=0):

        for key, val in params.items(): setattr(self, key, val)
        self.input_act = 1
        self.idx = idx

        if (self.old_actions_in_input == True):
            self.input_act += self.num_active_agents-1 

        self.policy = ActorCritic(params=params, input_size=self.input_act, output_size=self.action_size, \
            hidden_size=self.hidden_size_act).to(params.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_actor, eps=1e-5)
        self.memory = ExperienceReplayMemory(params, self.input_act)

        self.previous_action = torch.Tensor([1.])

        self.reset()

    def reset(self):
        self.memory.reset()
        self.memory.i = 0
        self.return_episode = 0.
        
    def get_action_and_value(self, state, action=None):
        return self.policy.get_action_and_value(state, action)
    
    def append_to_replay(self, s, a, l, r, d, v):
        self.memory._states[self.memory.i] = s
        self.memory._actions[self.memory.i] = a
        self.memory._logprobs[self.memory.i] = l
        self.memory._rewards[self.memory.i] = r
        self.memory._dones[self.memory.i] = d
        self.memory._values[self.memory.i] = v
        self.memory.i += 1
        self.return_episode =+ r

    def bootstrap(self, next_obs, next_done):
        #print("next_obs, next_done=",next_obs, next_done)
        values = self.memory._values
        dones = self.memory._dones
        rewards = self.memory._rewards
        #print("rewards=", rewards)
        with torch.no_grad():
            next_value = self.policy.get_value(next_obs).reshape(1, -1)
            self.advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_game_iterations)): # or num_steps
                if t == self.num_game_iterations - 1: # or num_steps
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                self.advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            self.returns = self.advantages + values
            #print("returns=", self.returns)
        
        self.flatten()

    def flatten(self):
        # flatten the batch
        self.b_obs = self.memory._states.reshape(-1, self.input_act)
        self.b_logprobs = self.memory._states.reshape(-1)
        self.b_actions = self.memory._actions.reshape(-1) #+ envs.single_action_space.shape)
        self.b_advantages = self.memory._advantages.reshape(-1)
        self.b_returns = self.returns.reshape(-1) #self.memory._returns.reshape(-1)
        self.b_values = self.memory._values.reshape(-1)

    def update(self):
        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for _ in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                #print("mb_inds=", mb_inds)

                #print("self.b_obs[mb_inds]=",self.b_obs[mb_inds])
                #print("self.b_actions.long()[mb_inds]=", self.b_actions.long()[mb_inds])
                _, newlogprob, entropy, newvalue = self.get_action_and_value(self.b_obs[mb_inds], self.b_actions.long()[mb_inds])
                logratio = newlogprob - self.b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = self.b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - self.b_returns[mb_inds]) ** 2
                    v_clipped = self.b_values[mb_inds] + torch.clamp(
                        newvalue - self.b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - self.b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - self.b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        y_pred, y_true = self.b_values.cpu().numpy(), self.b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y