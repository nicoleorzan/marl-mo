from src.environments import mo_epgg_v0 
from src.algos.MOReinforce import MOReinforce
from src.algos.Reinforce import Reinforce
import numpy as np
import optuna
from optuna.trial import TrialState
import torch
from optuna.storages import JournalStorage, JournalFileStorage
import wandb
from src.algos.normativeagent import NormativeAgent
from src.utils.social_norm import SocialNorm
from src.utils.utils import pick_agents_idxs, introspective_rewards
from src.experiments.params import setup_training_hyperparams

torch.autograd.set_detect_anomaly(True)

def define_agents(config):
    agents = {}
    for idx in range(config.n_agents):
        if (config.is_dummy[idx] == 0):
            if (config.num_objectives == 1):
                agents['agent_'+str(idx)] = MOReinforce(config, idx) 
            else:
                agents['agent_'+str(idx)] = MOReinforce(config, idx) 
        else: 
            agents['agent_'+str(idx)] = NormativeAgent(config, idx)
    return agents

def interaction_loop(config, parallel_env, active_agents, active_agents_idxs, social_norm, _eval=False, mf_input=None):
    # By default this is a training loop

    if (_eval == True):
        observations = parallel_env.reset(mf_input)
    else:
        observations = parallel_env.reset()
    
    #print("observations=",observations)
    #print("current mult=", parallel_env.current_multiplier)
    rewards_dict = {}
    actions_dict = {}
        
    states = {}; next_states = {}
    for idx_agent, agent in active_agents.items():
        other = active_agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
        if (agent.is_dummy == True): 
            next_states[idx_agent] = torch.cat((other.reputation, torch.Tensor([parallel_env.current_multiplier])))
        else: 
            if (config.reputation_enabled == 1):
                next_states[idx_agent] = torch.cat((other.reputation, observations[idx_agent]))
            else: 
                next_states[idx_agent] = observations[idx_agent]

    done = False
    for i in range(config.num_game_iterations):

        # state
        actions = {}; states = next_states; logprobs = {}
        for idx_agent, agent in active_agents.items():
            agent.state_act = states[idx_agent]
            #print("agent.state_act=", agent.state_act)
        #print("states=", states)
        
        # action
        for agent in parallel_env.active_agents:
            if (active_agents[agent].is_dummy == True): 
                a = active_agents[agent].select_action(_eval)
            else:
                a, logp = active_agents[agent].select_action(_eval)
                logprobs[agent] = logp
            actions[agent] = a
        #print("actions=", actions)
        #print("logprobs=",logprobs)

        # reward
        _, rewards, done, _ = parallel_env.step(actions)
        #print("rewards=", rewards)
        if (config.introspective == True):
            rewards = introspective_rewards(config, active_agents, parallel_env, rewards, actions)

        if (_eval==True):
            for ag_idx in active_agents_idxs:       
                if "agent_"+str(ag_idx) not in rewards_dict.keys():
                    rewards_dict["agent_"+str(ag_idx)] = [rewards["agent_"+str(ag_idx)]]
                    actions_dict["agent_"+str(ag_idx)] = [actions["agent_"+str(ag_idx)]]
                else:
                    rewards_dict["agent_"+str(ag_idx)].append(rewards["agent_"+str(ag_idx)])
                    actions_dict["agent_"+str(ag_idx)].append(actions["agent_"+str(ag_idx)])

        social_norm.save_actions(actions, active_agents_idxs)
        social_norm.rule09_binary_pgg(active_agents, active_agents_idxs, parallel_env.current_multiplier)

        # next state
        next_states = {}
        for idx_agent, agent in active_agents.items():
            other = active_agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
            if (agent.is_dummy == True): 
                next_states[idx_agent] = torch.cat((other.reputation, torch.Tensor([parallel_env.current_multiplier])))
            else: 
                if (config.reputation_enabled == 1):
                    next_states[idx_agent] = torch.cat((other.reputation, observations[idx_agent]))
                else: 
                    next_states[idx_agent] = observations[idx_agent]
           
        #if (_eval == False):
        # save iteration            
        for ag_idx, agent in active_agents.items():
            if (agent.is_dummy == False):
                agent.append_to_replay(states[ag_idx], actions[ag_idx], rewards[ag_idx], next_states[ag_idx], logprobs[ag_idx], done)
                agent.return_episode =+ rewards[ag_idx]

        if done:
            if (_eval == True):
                avg_reward = {}; avg_coop = {}
                for ag_idx, agent in active_agents.items():
                    avg_coop[ag_idx] = torch.mean(torch.stack(actions_dict[ag_idx]).float())
                    avg_reward[ag_idx] = torch.mean(torch.stack(rewards_dict[ag_idx]))
            break

    if (_eval == True):
        return avg_reward, avg_coop

def objective(args, repo_name, trial=None):

    all_params = setup_training_hyperparams(args, trial)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=args.wandb_mode)
    config = wandb.config
    print("config=", config)

    # define env
    parallel_env = mo_epgg_v0.parallel_env(config)

    # define agents
    agents = define_agents(config)

    # define social norm
    social_norm = SocialNorm(config, agents)
    
    #### TRAINING LOOP
    coop_agents_mf = {}; rew_agents_mf = {}
    for epoch in range(config.num_epochs):
        #print("\n==========>Epoch=", epoch)

        # pick a pair of agents
        active_agents_idxs = pick_agents_idxs(config)
        active_agents = {"agent_"+str(key): agents["agent_"+str(key)] for key, _ in zip(active_agents_idxs, agents)}

        [agent.reset() for _, agent in active_agents.items()]

        parallel_env.set_active_agents(active_agents_idxs)

        # TRAIN
        #print("\nTRAIN")
        interaction_loop(config, parallel_env, active_agents, active_agents_idxs, social_norm, _eval=False)

        # update agents
        losses = {}
        #print("\n\nUPDATE")
        for ag_idx, agent in active_agents.items():
            if (agent.is_dummy == True): 
                losses[ag_idx] = agent.update()
            else:
                if (config.num_objectives == 1):
                    losses[ag_idx] = agent.update_reward()
                else:
                    losses[ag_idx] = agent.update_mo()
               
        # EVAL
        #print("\nEVAL")
        for mf_input in config.mult_fact:
            [agent.reset() for _, agent in active_agents.items()]
            avg_rew, avg_coop = interaction_loop(config, parallel_env, active_agents, active_agents_idxs, social_norm, True, mf_input)
            avg_coop_tot = torch.mean(torch.stack([cop_val for _, cop_val in avg_coop.items()]))

            avg_rep = np.mean([agent.reputation[0] for _, agent in agents.items() if (agent.is_dummy == False)])
            measure = avg_rep
            coop_agents_mf[mf_input] = avg_coop
            rew_agents_mf[mf_input] = avg_rew

        dff_coop_per_mf = dict(("avg_coop_mf"+str(mf), torch.mean(torch.stack([ag_coop for _, ag_coop in coop_agents_mf[mf].items()]))) for mf in config.mult_fact)
        dff_rew_per_mf = dict(("avg_rew_mf"+str(mf), torch.mean(torch.stack([ag_coop for _, ag_coop in rew_agents_mf[mf].items()]))) for mf in config.mult_fact)

        prob = {}
        if (config.num_objectives == 1 ):
            if (config.reputation_enabled == 0): 
                for ag_idx, agent in agents.items():
                    if (agent.is_dummy == False):
                        if ag_idx not in prob:
                            prob[ag_idx] = torch.zeros(len(config.mult_fact), 2) # mult fact, poss actions
                            possible_states = torch.stack([torch.Tensor([mf]) for _, mf in enumerate(config.mult_fact)])
                            prob[ag_idx] = agent.read_distrib_no_reputation(possible_states,len(config.mult_fact)).detach()
            else:
                possible_reputations = [0., 1.]
                for ag_idx, agent in agents.items():
                    if (agent.is_dummy == False):
                        if ag_idx not in prob:
                            prob[ag_idx] = torch.zeros(2, len(config.mult_fact), 2) # mult fact, poss rep, poss actions
                        for rep in possible_reputations:
                            possible_states = torch.stack([torch.Tensor([rep, mf]) for _, mf in enumerate(config.mult_fact)])
                            prob[ag_idx][int(rep),:,:] = agent.read_distrib(possible_states,len(config.mult_fact)).detach()
            stacked = torch.stack([val for _, val in prob.items()])
            avg_distrib = torch.mean(stacked, dim=0)

        if (config.optuna_):
            trial.report(measure, epoch)
            
            if trial.should_prune():
                print("is time to pruneee")
                wandb.finish()
                raise optuna.exceptions.TrialPruned()

        if (config.wandb_mode == "online" and float(epoch)%30. == 0.):
            for ag_idx, agent in active_agents.items():
                if (agent.is_dummy == False):
                    df_avg_coop = {ag_idx+"avg_coop": avg_coop[ag_idx]}
                    df_avg_rew = {ag_idx+"avg_rew": avg_rew[ag_idx]}
                    df_loss = {ag_idx+"loss": losses[ag_idx]}
                    df_agent = {**{
                        ag_idx+"_reputation": agent.reputation,
                        'epoch': epoch}, 
                        **df_avg_coop, **df_avg_rew, **df_loss#, **df_distrib
                        }
                else:
                    df_avg_coop = {ag_idx+"dummy_avg_coop": avg_coop[ag_idx]}
                    df_avg_rew = {ag_idx+"dummy_avg_rew": avg_rew[ag_idx]}
                    df_agent = {**{
                        ag_idx+"dummy_reputation": agent.reputation,
                        'epoch': epoch}, 
                        **df_avg_coop, **df_avg_rew
                        }
                if ('df_agent' in locals() ):
                    wandb.log(df_agent, step=epoch, commit=False)

            dff = {
                "epoch": epoch,
                "avg_rep": avg_rep,
                "avg_rew_time": measure,
                "avg_coop_from_agents": avg_coop_tot,
                "weighted_average_coop": torch.mean(torch.stack([avg_i for _, avg_i in avg_rew.items()])) # only on the agents that played, of course
                }
            if (config.non_dummy_idxs != []): 
                    dff = {**dff, **dff_coop_per_mf, **dff_rew_per_mf}
                
            if (config.num_objectives == 1):
                if (config.reputation_enabled == 0): 
                    dff_prob00 = {"avg_prob["+str(mf)+",0]": avg_distrib[idx_m,0] for idx_m, mf in enumerate(config.mult_fact) }
                    dff_prob01 = {"avg_prob["+str(mf)+",1]": avg_distrib[idx_m,1] for idx_m, mf in enumerate(config.mult_fact) }
                    dff_prob10 = dict(())
                    dff_prob11 = dict(())
                else:
                    dff_prob00 = {"avg_prob[0,"+str(mf)+",0]": avg_distrib[0,idx_m,0] for idx_m, mf in enumerate(config.mult_fact) }
                    dff_prob01 = {"avg_prob[0,"+str(mf)+",1]": avg_distrib[0,idx_m,1] for idx_m, mf in enumerate(config.mult_fact) }
                    dff_prob10 = {"avg_prob[1,"+str(mf)+",0]": avg_distrib[1,idx_m,0] for idx_m, mf in enumerate(config.mult_fact) }
                    dff_prob11 = {"avg_prob[1,"+str(mf)+",1]": avg_distrib[1,idx_m,1] for idx_m, mf in enumerate(config.mult_fact) }
                dff = {**dff, **dff_prob00, **dff_prob01, **dff_prob10, **dff_prob11}
            
            wandb.log(dff,
                step=epoch, commit=True)

        if (epoch%10 == 0):
            print("\nEpoch : {} \t Measure: {} ".format(epoch, measure))
            print("avg_rew=", {ag_idx:avg_i for ag_idx, avg_i in avg_rew.items()})
            print("coop_agents_mf=",coop_agents_mf)
            print("dff_coop_per_mf=",dff_coop_per_mf)
    
    wandb.finish()
    return measure


def train_reinforce(args):

    unc_string = "no_unc_"
    if (args.uncertainties.count(0.) != args.n_agents):
        unc_string = "unc_"

    repo_name = "MO-EPGG_"+ str(args.n_agents) + "agents_" + \
        unc_string + args.algorithm + "_mf" + str(args.mult_fact) + \
        "_rep" + str(args.reputation_enabled)

    print("repo_name=", repo_name)

    # If optuna, then optimize or get best params. Else use given params
    if (args.optuna_ == 0):
        objective(args, repo_name)
    else:
        func = lambda trial: objective(args, repo_name, trial)

        # sql not optimized for paralel sync
        journal_name = repo_name + "_binary_"+str(args.binary_reputation)

        storage = JournalStorage(JournalFileStorage("optuna-journal"+journal_name+".log"))

        study = optuna.create_study(
            study_name=repo_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
            n_startup_trials=0, n_warmup_steps=40, interval_steps=3
            )
        )

        if (args.optimize):
            study.optimize(func, n_trials=100, timeout=1000)

        else:
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            
            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")

            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            
            print("Running with best params:")
            objective(args, repo_name+"_BEST", study.best_trial)