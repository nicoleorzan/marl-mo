from src.environments import mo_epgg_v0
from src.algos.MoDQN_sa import MoDQN
import numpy as np
import optuna
from optuna.trial import TrialState
import torch
from optuna.storages import JournalStorage, JournalFileStorage
import wandb
from itertools import product
import random
from src.experiments.experiments_sa.params_sa import setup_training_hyperparams

N_AGENTS_ACTIONS = 2

def pick_agents_idxs(config):

    possible_agents = [i for i in range(config.n_agents)]
    active_agents_idxs = []
    for _ in range(config.n_active_agents):
        ag_idx = random.sample(possible_agents, 1)[0] 
        active_agents_idxs.append(ag_idx)
        possible_agents.remove(ag_idx)
    return active_agents_idxs

torch.autograd.set_detect_anomaly(True)

def to_actions(config, possibilities, a, _id):
    actions = {}
    for i in range(config.n_active_agents):
        actions['agent_'+str(_id[i])] = torch.Tensor([possibilities[int(a)][i]])
    return actions


def interaction_loop(config, agent, parallel_env, active_agents_idxs, _eval=False, mf_input=None):
    # By default this is a training loop

    if (_eval == True):
        observations = parallel_env.reset(mf_input)
    else:
        observations = parallel_env.reset()

    #print("observations=",observations)
    #print('active_agents_idxs=',active_agents_idxs)
    rewards_dict = {}
    actions_dict = {}
    active_agents_idxs_n = [i/config.n_agents for i in active_agents_idxs]
    #print("active_agents_idxs_n=",active_agents_idxs_n)

    next_state = torch.zeros(config.n_active_agents+1)
    next_state[0] = observations['agent_'+str(active_agents_idxs[0])]
    for i in range(0,config.n_active_agents): 
        next_state[i+1] = torch.Tensor([active_agents_idxs_n[i]])
    #next_state = torch.unsqueeze(next_state,0)

    done = False
    possibilities = list(product(range(0,2), repeat=config.n_active_agents))

    for i in range(config.num_game_iterations):

        actions = {}; state = next_state
        #print("state=", state)
        
        # action
        action = agent.select_action(state,_eval)
        #print("action=", action)
        actions = to_actions(config, possibilities, action, active_agents_idxs)
        #print("actions=", actions)

        # reward
        _, rewards, done, _ = parallel_env.step(actions)
        #print("rewards=", rewards)

        if (_eval==True):
            for ag_idx in active_agents_idxs:       
                if "agent_"+str(ag_idx) not in rewards_dict.keys():
                    rewards_dict["agent_"+str(ag_idx)] = [rewards["agent_"+str(ag_idx)]]
                    actions_dict["agent_"+str(ag_idx)] = [actions["agent_"+str(ag_idx)]]
                else:
                    rewards_dict["agent_"+str(ag_idx)].append(rewards["agent_"+str(ag_idx)])
                    actions_dict["agent_"+str(ag_idx)].append(actions["agent_"+str(ag_idx)])
        
        # next state
        next_state = torch.zeros(config.n_active_agents+1)
        next_state[0] = observations['agent_'+str(active_agents_idxs[0])]
        for i in range(0,config.n_active_agents): 
            next_state[i+1] = torch.Tensor([active_agents_idxs_n[i]])
        #print("next state=", next_state)

        if (_eval == False):
            # save iteration
            agent.append_to_replay(state, action, rewards, next_state, done, active_agents_idxs)

        if done:
            if (_eval == True):
                avg_reward = {}; avg_coop = {}; scal_func = {}
                scal_func_tens = torch.zeros(config.n_active_agents); k=0
                for ag_idx in active_agents_idxs:
                    #print("actions_dict[ag_idx])=",actions_dict)
                    avg_coop["agent_"+str(ag_idx)] = torch.mean(torch.stack(actions_dict["agent_"+str(ag_idx)]).float())
                    # computing avg_reward for every objective (expectation)
                    avg_reward[ag_idx] = torch.mean(torch.stack(rewards_dict["agent_"+str(ag_idx)]), dim=0).unsqueeze(1)
                    #avg_reward[ag_idx] = torch.mean(torch.stack(rewards_dict), dim=0).unsqueeze(1)
                    #print("rewards_dict=", rewards_dict)
                    #print("avg_reward=",avg_reward[ag_idx])
                    #computing scalarization function, after expectation (SER)
                    if (config.num_objectives > 1):
                        #print("avg_reward[ag_idx]=",avg_reward[ag_idx])
                        #print("agent.w_agents=", agent.w_agents)
                        scal_func[ag_idx] = agent.scal_func_agents[ag_idx](avg_reward[ag_idx], agent.w_agents)
                        scal_func_tens[k] = agent.scal_func_agents[ag_idx](avg_reward[ag_idx], agent.w_agents)
                        k+=1
                #print("scal_func=", scal_func)
                #print("scal_func_tens=", scal_func_tens)
                final_scal_value = agent.scal_func(scal_func_tens, agent.w, _dim=0)
                #print("final_scal_value=",final_scal_value)
            break

    if (_eval == True):
        return avg_reward, avg_coop, scal_func, final_scal_value

def objective(args, repo_name, trial=None):

    all_params = setup_training_hyperparams(args, trial)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=args.wandb_mode)
    config = wandb.config
    print("config=", config)

    # define envupdate_
    parallel_env = mo_epgg_v0.parallel_env(config)

    # define agents
    agent = MoDQN(config) 

    # define social norm
    #social_norm = SocialNorm(config, agent)
    
    #### TRAINING LOOP

    coop_agents_mf = {}; rew_agents_mf = {}; scal_func_agents_mf ={}; scal_func_mf ={}
    for epoch in range(config.num_epochs):
        #print("\n==========>Epoch=", epoch)

        # pick a pair of agents
        active_agents_idxs = pick_agents_idxs(config)
        #print("active_agents_idxs=",active_agents_idxs)
        #active_agents = {"agent_"+str(key): agent["agent_"+str(key)] for key, _ in zip(active_agents_idxs, agent)}

        #[agent.reset() for _, agent in active_agents.items()]

        parallel_env.set_active_agents(active_agents_idxs)

        # TRAIN
        #print("\n\nTRAIN")
        interaction_loop(config, agent, parallel_env, active_agents_idxs,  _eval=False)

        # update agents
        #print("\n\nUPDATE!!")
        agent.update(epoch)

        # evaluation step
        #print("\n\nEVAL")
        for mf_input in config.mult_fact:
            #print("mf=", mf_input)
            avg_rew, avg_coop, scal_func, final_scal_value = interaction_loop(config, agent, parallel_env, active_agents_idxs, True, mf_input)
            avg_coop_tot = torch.mean(torch.stack([cop_val for _, cop_val in avg_coop.items()]))
            #print("==>avg_rew over loop=", avg_rew)
            #print("avg_coop=", avg_coop)
            coop_agents_mf[mf_input] = avg_coop
            rew_agents_mf[mf_input] = avg_rew
            scal_func_agents_mf[mf_input] = scal_func
            scal_func_mf[mf_input] = final_scal_value
        #print("coop_agents_mf=",coop_agents_mf)

        dff_coop_per_mf = dict(("avg_coop_mf"+str(mf), torch.mean(torch.stack([ag_coop for _, ag_coop in coop_agents_mf[mf].items()]))) for mf in config.mult_fact)
        dff_rew_per_mf = dict(("avg_rew_mf"+str(mf), torch.mean(torch.stack([ag_coop for _, ag_coop in rew_agents_mf[mf].items()]))) for mf in config.mult_fact)

        if (config.wandb_mode == "online" and float(epoch)%30. == 0.):
            for ag_idx in active_agents_idxs:
                df_avg_coop = dict(("agent_"+str(ag_idx)+"avg_coop_mf"+str(mf_input), coop_agents_mf[mf_input]["agent_"+str(ag_idx)]) for mf_input in config.mult_fact)
                df_scal_func_agents = {}
                if (config.num_objectives > 1):
                    df_scal_func_agents = dict(("agent_"+str(ag_idx)+"avg_scal_func"+str(mf_input), scal_func_agents_mf[mf_input][ag_idx]) for mf_input in config.mult_fact)
                df_avg_rew = {}
                for obj_idx in range(config.num_objectives):
                    df_rews_tmp = dict(("agent_"+str(ag_idx)+"rew_mf"+str(mf_input)+"_obj"+str(obj_idx), rew_agents_mf[mf_input][ag_idx][obj_idx]) for mf_input in config.mult_fact)
                    df_avg_rew = {**df_avg_rew, **df_rews_tmp}
                df_agent = {**{
                    'epoch': epoch}, 
                    **df_avg_coop, **df_avg_rew, **df_scal_func_agents
                    }
                if ('df_agent' in locals() ):
                    wandb.log(df_agent, step=epoch, commit=False)
            df_scal_func = {}
            if (config.num_objectives > 1):
                df_scal_func = dict(("avg_scal_func"+str(mf_input), scal_func_agents_mf[mf_input]) for mf_input in config.mult_fact)
                
            dff = {
                "epoch": epoch,
                "avg_coop_from_agents": avg_coop_tot,
                "weighted_average_coop": torch.mean(torch.stack([avg_i for _, avg_i in avg_rew.items()])) # only on the agents that played, of course
                }
            if (config.non_dummy_idxs != []): 
                dff = {**dff, **dff_coop_per_mf, **dff_rew_per_mf, **df_scal_func}
            wandb.log(dff,
                step=epoch, commit=True)

        if (epoch%10 == 0):
            print("\nEpoch : {} ".format(epoch))
            print("avg_rew=", {ag_idx:avg_i for ag_idx, avg_i in avg_rew.items()})
            #print("avg_coop_tot=", avg_coop_tot)
            print("coop_agents_mf=",coop_agents_mf)
            print("dff_coop_per_mf=",dff_coop_per_mf)
    
    wandb.finish()
    return 0


def train_dqn(args):

    unc_string = "no_unc_"
    if (args.uncertainties.count(0.) != args.n_agents):
        unc_string = "unc_"

    repo_name = "MO-EPGG-SA_"+ str(args.n_agents) + "agents_" + \
        unc_string + "_mf" + str(args.mult_fact) + \
        "_rep" + str(args.reputation_enabled)
    
    #if (args.addition != ""):
    #    repo_name += "_"+ str(args.addition)
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