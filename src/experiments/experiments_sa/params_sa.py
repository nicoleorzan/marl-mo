import torch

RANDOM_BASELINE = False
N_ACTIONS_AGENTS = 2

DEVICE = torch.device('cpu')
if(torch.cuda.is_available()): 
    DEVICE = torch.device('cuda:0') 
    torch.cuda.empty_cache()

def setup_training_hyperparams(args, trial):

    all_params = {}

    game_params = dict(
        n_agents = args.n_agents,
        n_active_agents = args.n_active_agents,
        n_actions_agents = N_ACTIONS_AGENTS,
        mult_fact = args.mult_fact,
        num_epochs = args.num_epochs,
        num_game_iterations = args.num_game_iterations,
        num_objectives = args.num_objectives,
        uncertainties = args.uncertainties,
        coins_value = args.coins_value,
        wandb_mode = args.wandb_mode,
        proportion_dummy_agents = args.proportion_dummy_agents,
        get_index = False,
        action_size = N_ACTIONS_AGENTS, #2**args.n_active_agents,
        random_baseline = RANDOM_BASELINE,
        embedding_dim = 1,
        optuna_ = args.optuna_,
        reputation_enabled = args.reputation_enabled,
        introspective = args.introspective,
        rule = args.rule,
        reputation_assignment = args.reputation_assignment,
        device = DEVICE,
        scalarization_function = args.scalarization_function,
        mf_from_interval = args.mf_from_interval,
        print_step = args.print_step,
        weights = args.weights,
        weights_agents = args.weights_agents,
        agents_functions = args.agents_funcs
    )

    obs_size = args.n_active_agents+1  # mf, idx first agent, idx second agent
    algo_params = dict(
        obs_size = obs_size, # mult factor and reputation of opponent
        gamma = 0.99,
        chi = 0.001,
        epsilon = args.epsilon_dqn,
        memory_size = 500,
        n_hidden_act = 1,
        #freq_counts = args.freq_counts,
        hidden_size_act = 4,
        dqn_activation_function = args.dqn_activation_function,
        lr_actor = args.lr_dqn,
        decayRate = 0.999,
        target_net_update_freq = 30,
        alpha = 0.1, # introspection level
        decaying_epsilon = True
    )

    n_dummy = int(args.proportion_dummy_agents*args.n_agents)
    is_dummy = list(reversed([1 if i<n_dummy else 0 for i in range(args.n_agents) ]))
    non_dummy_idxs = [i for i,val in enumerate(is_dummy) if val==0]

    all_params = {**all_params, **game_params, **algo_params,  "n_dummy":n_dummy, "is_dummy":is_dummy, "non_dummy_idxs":non_dummy_idxs}
    print("all_params=", all_params)

    return all_params