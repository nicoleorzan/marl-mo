from environments import mo_epgg_v0 
from algos.MOReinforce_SER_new import MOReinforce
import numpy as np
import torch
import wandb
from utils.utils import pick_agents_idxs, introspective_rewards
from experiments.params import setup_training_hyperparams

torch.autograd.set_detect_anomaly(True)


def define_agents(config):
    agents = {}
    for idx in range(config.n_agents):
        agents['agent_'+str(idx)] = MOReinforce(config, idx) 
    return agents

def interaction_loop(config, parallel_env, active_agents, active_agents_idxs, _eval=False, mf_input=None):
    # By default this is a training loop

    batch_s = config.batch_size
    if (_eval == True):
        batch_s = 1
        rewards_dict = {}; actions_dict = {}; distrib_dict = {}
        avg_reward = {}; avg_coop = {}; avg_distrib = {}; scal_func = {}

    for i_batch in range(batch_s):
        #print("\ni_batch=", i_batch)
        for _,agent in active_agents.items():
            agent.memory.i = 0

        if (_eval == True):
            observations = parallel_env.reset(mf_input)
        else:
            observations = parallel_env.reset()
        #print("parallel_env.current_multiplier=", parallel_env.current_multiplier)

        states = {}; next_states = {}
        for idx_agent, agent in active_agents.items():
            next_states[idx_agent] = observations[idx_agent]
            if (config.old_actions_in_input == True):
                others_acts = torch.stack([torch.Tensor([active_agents["agent_"+str(_id)].previous_action ]) for _id in active_agents_idxs if "agent_"+str(_id)!=idx_agent], dim=1).squeeze(0)
                next_states[idx_agent] = torch.cat((next_states[idx_agent], others_acts))

        done = False
        
        for i in range(config.num_game_iterations):
            #print("i=",i)

            actions = {}; states = next_states; logprobs = {}; critic = {}
            #print("states=", states)
            for idx_agent, agent in active_agents.items():
                a, logp, dist = active_agents[idx_agent].select_action(states[idx_agent], _eval)
                logprobs[idx_agent] = logp
                actions[idx_agent] = a
                critic[idx_agent] = dist
    
            #print("actions=", actions)
            #print("logprobs=",logprobs)

            _, rewards, done, _ = parallel_env.step(actions)
            #print("rewards=", rewards)
          
            #print("rewards=", rewards)
            if (config.introspective == True):
                rewards = introspective_rewards(config, observations, active_agents, parallel_env, rewards, actions)

            if (_eval==True):
                for idx_agent, agent in active_agents.items():
                    if "agent_"+str(idx_agent) not in rewards_dict.keys():
                        rewards_dict[idx_agent] = [rewards[idx_agent]]
                        actions_dict[idx_agent] = [actions[idx_agent]]
                        distrib_dict[idx_agent] = [active_agents[idx_agent].get_action_distribution(states[idx_agent])]
                    else:
                        rewards_dict[idx_agent].append(rewards[idx_agent])
                        actions_dict[idx_agent].append(actions[idx_agent])
                        distrib_dict[idx_agent].append(active_agents[idx_agent].get_action_distribution(states[idx_agent]))

            next_states = {}
            for idx_agent, agent in active_agents.items():
                next_states[idx_agent] = observations[idx_agent]
                if (config.old_actions_in_input == True):
                    others_acts = torch.stack([torch.Tensor([active_agents["agent_"+str(_id)].previous_action ]) for _id in active_agents_idxs if "agent_"+str(_id)!=idx_agent], dim=1).squeeze(0)
                    next_states[idx_agent] = torch.cat((next_states[idx_agent], others_acts))
            #print("next_states=", next_states)   

            if (_eval == False):
                # save iteration
                for idx_agent, agent in active_agents.items():
                    agent.append_to_replay(i_batch, states[idx_agent], actions[idx_agent], rewards[idx_agent], next_states[idx_agent], logprobs[idx_agent], critic[idx_agent], done)
                    agent.return_episode =+ rewards[idx_agent]

            if done:
                if (_eval == True):
                    for idx_agent, agent in active_agents.items():
                        avg_coop[idx_agent] = torch.mean(torch.stack(actions_dict[idx_agent]).float())
                        # computing avg_reward for every objective (expectation)
                        avg_reward[idx_agent] = torch.mean(torch.stack(rewards_dict[idx_agent]), dim=0)#.unsqueeze(1)
                        avg_distrib[idx_agent] = torch.mean(torch.stack(distrib_dict[idx_agent]), dim=0).unsqueeze(1)
                        if (config.num_objectives > 1):
                            #print("avg_reward[idx_agent]=",avg_reward[idx_agent], avg_reward[idx_agent].shape)
                            #print("other=", avg_reward[idx_agent].reshape(1,config.num_objectives), avg_reward[idx_agent].reshape(1,config.num_objectives).shape)
                            #print("avg_reward[idx_agent].reshape(1,config.num_objectives=",avg_reward[idx_agent].reshape(1,config.num_objectives))
                            #print("avg_reward[idx_agent].reshape(1,config.num_objectives)=",avg_reward[idx_agent].reshape(1,config.num_objectives))
                            scal_func[idx_agent] = agent.beta_utility(avg_reward[idx_agent])
                break

    if (_eval == True):
        return avg_reward, avg_coop, avg_distrib, scal_func

def objective(args, repo_name, trial=None):

    all_params = setup_training_hyperparams(args, trial)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=args.wandb_mode)
    config = wandb.config
    print("config=", config)

    # define env
    parallel_env = mo_epgg_v0.parallel_env(config)

    # define agents
    agents = define_agents(config)

    #### TRAINING LOOP
    coop_agents_mf = {}; rew_agents_mf = {}; scal_func_mf = {}; distrib_agents_mf = {}
    
    for epoch in range(config.num_epochs):
        #print("\n==========>Epoch=", epoch)

        # pick a group of agents
        active_agents_idxs = pick_agents_idxs(config)
        active_agents = {"agent_"+str(key): agents["agent_"+str(key)] for key, _ in zip(active_agents_idxs, agents)}
        #print("active_agents_idxs=", active_agents_idxs)

        [agent.reset() for _, agent in active_agents.items()]

        parallel_env.set_active_agents(active_agents_idxs)

        # TRAIN
        #print("\n\nTRAIN")
        interaction_loop(config, parallel_env, active_agents, active_agents_idxs, _eval=False)

        # update agents
        #print("\n\nUPDATE!!")
        losses = {}
        for idx_agent, agent in active_agents.items():
            losses[idx_agent] = agent.update_mo_ser()
               
        # evaluation step
        #print("\n\nEVAL")
        if (float(epoch)%float(config.print_step) == 0.):
            for mf_input in config.mult_fact:
                avg_rew, avg_coop, avg_distrib, scal_func = interaction_loop(config, parallel_env, active_agents, active_agents_idxs, True, mf_input)
                #print('avg_rew, avg_coop, scal_func=',avg_rew, avg_coop, scal_func)
                avg_coop_tot = torch.mean(torch.stack([cop_val for _, cop_val in avg_coop.items()]))
                avg_rep = np.mean([agent.reputation[0] for _, agent in agents.items()])
                coop_agents_mf[mf_input] = avg_coop
                distrib_agents_mf[mf_input] = avg_distrib
                rew_agents_mf[mf_input] = avg_rew
                scal_func_mf[mf_input] = scal_func

            #print("distrib_agents_mf",distrib_agents_mf)
            #a = torch.stack([ag_dis for _, ag_dis in distrib_agents_mf[0.5].items()])
            #a = a.view(2,2)
            #print("mean=", torch.mean(a, dim=0).squeeze()) # ora piglio solo la prob cooperazione, menainng axis [0]
            #print("mean=", torch.mean(a, dim=0).squeeze()[0]) # ora piglio solo la prob cooperazione, menainng axis [0]
            dff_distrib_per_mf = dict(("avg_distrib_mf"+str(mf), torch.mean(torch.stack([ag_dis for _, ag_dis in distrib_agents_mf[mf].items()]), dim=0).squeeze()[1]) for mf in config.mult_fact)
            #print("dff_distrib_per_mf",dff_distrib_per_mf)

            #print("torch.mean(torch.stack([ag_dis for _, ag_dis in distrib_agents_mf[mf].items()]))) for mf in config.mult_fact)",torch.mean(torch.stack([ag_dis for _, ag_dis in distrib_agents_mf[0.5].items()])))
            dff_coop_per_mf = dict(("avg_coop_mf"+str(mf), torch.mean(torch.stack([ag_coop for _, ag_coop in coop_agents_mf[mf].items()]))) for mf in config.mult_fact)
            dff_rew_per_mf = dict(("avg_rew_mf"+str(mf), torch.mean(torch.stack([ag_coop for _, ag_coop in rew_agents_mf[mf].items()]))) for mf in config.mult_fact)

        if (config.wandb_mode == "online" and float(epoch)%float(config.print_step) == 0.):
            #print("logging")
            for idx_agent, agent in active_agents.items():
                df_avg_coop = {idx_agent+"avg_coop": avg_coop[idx_agent]}
                df_scal_func = {}
                if (config.num_objectives > 1):
                    df_scal_func = dict((idx_agent+"avg_scal_func"+str(mf_input), scal_func_mf[mf_input][idx_agent]) for mf_input in config.mult_fact)
                df_avg_rew = {}
                for obj_idx in range(config.num_objectives):
                    df_rews_tmp = dict((idx_agent+"rew_mf"+str(mf_input)+"_obj"+str(obj_idx), rew_agents_mf[mf_input][idx_agent][obj_idx]) for mf_input in config.mult_fact)
                    df_avg_rew = {**df_avg_rew, **df_rews_tmp}
                df_loss = {idx_agent+"loss": losses[idx_agent]}
            dff = {
                "epoch": epoch,
                "avg_rep": avg_rep,
                "avg_coop_from_agents": avg_coop_tot,
                "weighted_average_coop": torch.mean(torch.stack([avg_i for _, avg_i in avg_rew.items()])) # only on the agents that played, of course
                }
            if (config.non_dummy_idxs != []): 
                dff = {**dff, **dff_coop_per_mf, **dff_distrib_per_mf, **dff_rew_per_mf}
            wandb.log(dff,
                step=epoch, commit=True)

        if (epoch%config.print_step == 0):
            print("\nEpoch : {}".format(epoch))
            #print("dff_rew_per_mf=", dff_rew_per_mf)
            #print("coop_agents_mf=",coop_agents_mf)
            #print("distrib_agents_mf=",distrib_agents_mf)
            print("prob cooperazione=",dff_distrib_per_mf)

            #print("agents[agent_0].get_action_distribution(torch.Tensor([[1., 1.0000]]))=",agents["agent_0"].get_action_distribution(torch.Tensor([[1., 1.0000]])))

    wandb.finish()


def train_ac(args):

    unc_string = "no_unc_"
    if (args.uncertainties.count(0.) != args.n_agents):
        unc_string = "unc_"

    repo_name = "ACTOR-CRITIC_NEW_MO-EPGG_"+ str(args.n_agents) + "agents_" + \
        unc_string + args.algorithm + "_mf" + str(args.mult_fact) + \
        "_rep" + str(args.reputation_enabled) + "_n_act_agents" + str(args.num_active_agents)
    
    print("repo_name=", repo_name)

    objective(args, repo_name)