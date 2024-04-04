from src.algos.PPO_cartpole import PPO
import numpy as np
import gym
import torch
import wandb
from src.experiments.cartpole.params_cartpole import setup_training_hyperparams

torch.autograd.set_detect_anomaly(True)
_print = 0
env = gym.make("CartPole-v1" )

def define_agents(config):
    agents = {}
    agents['agent_0'] = PPO(params=config, idx=0) 
    return agents

def interaction_loop(config, active_agents, active_agents_idxs, _eval=False):
    # By default this is a training loop

    next_obs = env.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs)

    #print("observations=",observations)
    rewards_dict = {}; actions_dict = {}
    ep_return = 0

    states = {}; next_states = {}
    for idx_agent, agent in active_agents.items():
        next_states[idx_agent] = next_obs

    done = False
    for i in range(config.num_steps):
        #print("step=",i)
        # state
        actions = {}; states = next_states; logprobs = {}; values = {}
        
        # action
        for ag_idx, agent in active_agents.items():
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(states[ag_idx])
                values[ag_idx] = value.flatten()
            actions[ag_idx] = action
            logprobs[ag_idx] = logprob
        for ag_idx, agent in active_agents.items():
            agent.previous_action = actions[ag_idx]

        # reward
        next_obs, reward, done, infos = env.step(actions["agent_0"].numpy())
        ep_return += reward

        if (_print):
            print("states=", states)
            print("actions=", actions)
            print("rewards=", reward)

        if (_eval==True):
            for ag_idx in active_agents_idxs:       
                if "agent_"+str(ag_idx) not in rewards_dict.keys():
                    rewards_dict["agent_"+str(ag_idx)] = [reward]
                    actions_dict["agent_"+str(ag_idx)] = [actions["agent_"+str(ag_idx)]]
                else:
                    rewards_dict["agent_"+str(ag_idx)].append(reward)
                    actions_dict["agent_"+str(ag_idx)].append(actions["agent_"+str(ag_idx)])

        if (_eval == False):
            for ag_idx, agent in active_agents.items():
                agent.append_to_replay(states[ag_idx], actions[ag_idx], logprobs[ag_idx], reward, done, values[ag_idx])

        # bootstrap value if not done
        if (_eval == False and done):
            for ag_idx, agent in active_agents.items():
                agent.bootstrap(next_states[ag_idx], done)

        if done:
            if (_eval == True):
                avg_reward = {}; avg_coop = {}; scal_func = {}
                for ag_idx, agent in active_agents.items():
                    avg_reward[ag_idx] = np.mean(rewards_dict[ag_idx])
            break

    return ep_return

def objective(args, repo_name, trial=None):

    all_params = setup_training_hyperparams(args, trial)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=args.wandb_mode)
    config = wandb.config
    print("config=", config)

    # TRY NOT TO MODIFY: seeding
    #np.random.seed(config.seed)
    #torch.manual_seed(config.seed)
    #torch.backends.cudnn.deterministic = config.torch_deterministic

    # define env

    # define agents
    agents = define_agents(config)

    #### TRAINING LOOP
    print("Num iterations=", config.num_iterations)
    for iteration in range(config.num_iterations):
        if (_print):
            print("\niteration=", iteration)

        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            lrnow = frac * config.lr_actor
            for agent_idx in agents:
                agents[agent_idx].optimizer.param_groups[0]["lr"] = lrnow

        # pick a group of agents
        active_agents = {"agent_0": agents["agent_0"]}

        [agent.reset() for _, agent in active_agents.items()]

        active_agents_idxs = [0]

        # TRAIN
        #print("\n\nTRAIN")
        interaction_loop(config, active_agents, active_agents_idxs, _eval=False)

        # update agents
        #print("\n\nUPDATE!!")
        losses = {}
        for ag_idx, agent in active_agents.items():
            losses[ag_idx] = agent.update()

        # evaluation step
        #print("\n\nEVAL")
        if (float(iteration)%float(config.print_step) == 0.):
            ep_return = interaction_loop(config, active_agents, active_agents_idxs, True)

        if (config.wandb_mode == "online" and float(iteration)%float(config.print_step) == 0.):
            dff = {
                "epoch": iteration,
                "ep_return": ep_return,
                "lr_actor": lrnow,
                }
            wandb.log(dff,
                step=iteration, commit=True)

        if (iteration%config.print_step == 0):
            print("\nEpoch : {}".format(iteration))
            print("ep_return=", ep_return)
    
    wandb.finish()
    return 


def train_ppo_single_obj(args):

    repo_name = "CARTPOLE"
    
    print("repo_name=", repo_name)

    objective(args, repo_name)
