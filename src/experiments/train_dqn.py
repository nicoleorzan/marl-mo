from environments import mo_epgg_v0
from algos.MoDQN import MoDQN
from algos.DQN import DQN
import numpy as np
import optuna
from optuna.trial import TrialState
import torch
from optuna.storages import JournalStorage, JournalFileStorage
import wandb
from algos.normativeagent import NormativeAgent
from utils.social_norm import SocialNorm
from utils.utils import pick_agents_idxs, introspective_rewards
from experiments.params import setup_training_hyperparams

torch.autograd.set_detect_anomaly(True)

def define_agents(config):
    agents = {}
    for idx in range(config.n_agents):
        if (config.is_dummy[idx] == 0):
            #if (config.num_objectives == 1):
            #    agents['agent_'+str(idx)] = DQN(config, idx) 
            #else:
            agents['agent_'+str(idx)] = MoDQN(config, idx) 
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
        if (config.old_actions_in_input == True):
            others_acts = torch.stack([torch.Tensor([active_agents["agent_"+str(_id)].previous_action ]) for _id in active_agents_idxs if "agent_"+str(_id)!=idx_agent], dim=1).squeeze(0)
            next_states[idx_agent] = torch.cat((next_states[idx_agent], others_acts))

    done = False
    for i in range(config.num_game_iterations):
        # state
        actions = {}; states = next_states
        
        # action
        for idx_agent, agent in active_agents.items():
            a = active_agents[idx_agent].select_action(states[idx_agent],_eval)
            actions[idx_agent] = a
        for ag_idx, agent in active_agents.items():
            agent.previous_action = actions[ag_idx]
        #print("actions=", actions)

        # reward
        _, rewards, done, _ = parallel_env.step(actions)
        #print("rewards=", rewards)
        if (config.introspective == True):
            rewards = introspective_rewards(config, observations, active_agents, parallel_env, rewards, actions)

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
            if (config.old_actions_in_input == True):
                others_acts = torch.stack([torch.Tensor([active_agents["agent_"+str(_id)].previous_action ]) for _id in active_agents_idxs if "agent_"+str(_id)!=idx_agent], dim=1).squeeze(0)
                next_states[idx_agent] = torch.cat((next_states[idx_agent], others_acts))

        if (_eval == False):
            # save iteration
            for ag_idx, agent in active_agents.items():
                if (agent.is_dummy == False):
                    agent.append_to_replay(states[ag_idx], actions[ag_idx], rewards[ag_idx], next_states[ag_idx], done)
                    agent.return_episode =+ rewards[ag_idx]

        if done:
            if (_eval == True):
                avg_reward = {}; avg_coop = {}; scal_func = {}
                for ag_idx, agent in active_agents.items():
                    avg_coop[ag_idx] = torch.mean(torch.stack(actions_dict[ag_idx]).float())
                    # computing avg_reward for every objective (expectation)
                    avg_reward[ag_idx] = torch.mean(torch.stack(rewards_dict[ag_idx]), dim=0).unsqueeze(1)
                    #computing scalarization function, after expectation (SER)
                    if (config.num_objectives > 1):
                        scal_func[ag_idx] = agent.scal_func(avg_reward[ag_idx].unsqueeze(0), agent.w)
                    #print("scal func=", scal_func)
            break

    if (_eval == True):
        return avg_reward, avg_coop, scal_func

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
    coop_agents_mf = {}; rew_agents_mf = {}; scal_func_mf ={}
    for epoch in range(config.num_epochs):
        #print("\n==========>Epoch=", epoch)

        # pick a group of agents
        active_agents_idxs = pick_agents_idxs(config)
        active_agents = {"agent_"+str(key): agents["agent_"+str(key)] for key, _ in zip(active_agents_idxs, agents)}

        [agent.reset() for _, agent in active_agents.items()]

        parallel_env.set_active_agents(active_agents_idxs)

        # TRAIN
        #print("\n\nTRAIN")
        interaction_loop(config, parallel_env, active_agents, active_agents_idxs, social_norm, _eval=False)

        # update agents
        #print("\n\nUPDATE!!")
        losses = {}
        for ag_idx, agent in active_agents.items():
            losses[ag_idx] = agent.update(epoch)

        # evaluation step
        #print("\n\nEVAL")
        if (float(epoch)%float(config.print_step) == 0.):
            for mf_input in config.mult_fact:
                avg_rew, avg_coop, scal_func = interaction_loop(config, parallel_env, active_agents, active_agents_idxs, social_norm, True, mf_input)
                avg_coop_tot = torch.mean(torch.stack([cop_val for _, cop_val in avg_coop.items()]))
                avg_rep = np.mean([agent.reputation[0] for _, agent in agents.items() if (agent.is_dummy == False)])
                measure = avg_rep
                coop_agents_mf[mf_input] = avg_coop
                rew_agents_mf[mf_input] = avg_rew
                scal_func_mf[mf_input] = scal_func
            #print("rew_agents_mf=",rew_agents_mf)
            #print("scal_func_mf=",scal_func_mf)

            dff_coop_per_mf = dict(("avg_coop_mf"+str(mf), torch.mean(torch.stack([ag_coop for _, ag_coop in coop_agents_mf[mf].items()]))) for mf in config.mult_fact)
            dff_rew_per_mf = dict(("avg_rew_mf"+str(mf), torch.mean(torch.stack([ag_coop for _, ag_coop in rew_agents_mf[mf].items()]))) for mf in config.mult_fact)

            if (config.optuna_):
                trial.report(measure, epoch)
                
                if trial.should_prune():
                    print("is time to pruneee")
                    wandb.finish()
                    raise optuna.exceptions.TrialPruned()

        if (config.wandb_mode == "online" and float(epoch)%float(config.print_step) == 0.):
            for ag_idx, agent in active_agents.items():
                if (agent.is_dummy == False):
                    df_avg_coop = dict((ag_idx+"avg_coop_mf"+str(mf_input), coop_agents_mf[mf_input][ag_idx]) for mf_input in config.mult_fact)
                    df_scal_func = {}
                    if (config.num_objectives > 1):
                        df_scal_func = dict((ag_idx+"avg_scal_func"+str(mf_input), scal_func_mf[mf_input][ag_idx]) for mf_input in config.mult_fact)
                    df_avg_rew = {}
                    for obj_idx in range(config.num_objectives):
                        df_rews_tmp = dict((ag_idx+"rew_mf"+str(mf_input)+"_obj"+str(obj_idx), rew_agents_mf[mf_input][ag_idx][obj_idx]) for mf_input in config.mult_fact)
                        df_avg_rew = {**df_avg_rew, **df_rews_tmp}
                    df_loss = {ag_idx+"loss": losses[ag_idx]}
                    df_agent = {**{
                        ag_idx+"_reputation": agent.reputation,
                        ag_idx+"epsilon": active_agents[str(ag_idx)].epsilon,
                        'epoch': epoch}, 
                        **df_avg_coop, **df_avg_rew, **df_loss, **df_scal_func
                        }
                else:
                    df_avg_coop = {ag_idx+"dummy_avg_coop": avg_coop[ag_idx]}
                    df_scal_func = {}
                    if (config.num_objectives > 1):
                        df_scal_func = {ag_idx+"dummy_scal_func": scal_func_mf[ag_idx]}
                    df_avg_rew = {}
                    for obj_idx in range(config.num_objectives):
                        df_rews_tmp = dict((ag_idx+"rew_mf"+str(mf_input)+"_obj"+str(obj_idx), rew_agents_mf[mf_input][ag_idx][obj_idx]) for mf_input in config.mult_fact)
                        df_avg_rew = {**df_avg_rew, **df_rews_tmp}
                    df_agent = {**{
                        ag_idx+"dummy_reputation": agent.reputation,
                        'epoch': epoch}, 
                        **df_avg_coop, **df_avg_rew, **df_scal_func
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
            wandb.log(dff,
                step=epoch, commit=True)

        if (epoch%config.print_step == 0):
            print("\nEpoch : {} \t Measure: {} ".format(epoch, measure))
            print("avg_rew=", {ag_idx:avg_i for ag_idx, avg_i in avg_rew.items()})
            print("coop_agents_mf=",coop_agents_mf)
            print("dff_coop_per_mf=",dff_coop_per_mf)
    
    wandb.finish()
    return measure


def train_dqn(args):

    unc_string = "no_unc_"
    if (args.uncertainties.count(0.) != args.n_agents):
        unc_string = "unc_"

    repo_name = "OTHER_NEW_MO-EPGG_"+ str(args.n_agents) + "agents_" + \
        unc_string + args.algorithm + "_mf" + str(args.mult_fact) + \
        "_rep" + str(args.reputation_enabled) + "_n_act_agents" + str(args.num_active_agents)
    
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