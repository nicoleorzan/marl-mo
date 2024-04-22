import torch

ACTION_SIZE = 2
RANDOM_BASELINE = False

def setup_training_hyperparams(args, trial):

    all_params = {}

    game_params = dict(
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu"),
        n_agents = args.n_agents,
        mult_fact = args.mult_fact,
        num_epochs = args.num_epochs,
        num_game_iterations = args.num_game_iterations,
        batch_size = args.batch_size,
        num_active_agents = args.num_active_agents,
        num_objectives = args.num_objectives,
        uncertainties = args.uncertainties,
        algorithm = args.algorithm,
        coins_value = args.coins_value,
        wandb_mode = args.wandb_mode,
        proportion_dummy_agents = args.proportion_dummy_agents,
        get_index = False,
        action_size = ACTION_SIZE,
        random_baseline = RANDOM_BASELINE,
        embedding_dim = 1,
        optuna_ = args.optuna_,
        reputation_enabled = args.reputation_enabled,
        old_actions_in_input = args.old_actions_in_input,
        introspective = args.introspective,
        rule = args.rule,
        reputation_assignment = args.reputation_assignment,
        scalarization_function = args.scalarization_function,
        mf_from_interval = args.mf_from_interval,
        print_step = args.print_step,
        weights = args.weights,
        betas = args.betas,
        betas_from_distrib = args.betas_from_distrib,
        sigma_beta = args.sigma_beta,
        seed = 123
    )

    if (args.algorithm == "reinforce"):  
        num_hidden_a = 1
        hidden_size_a = 4
        obs_size = 2 # m factor and action of other
        algo_params = dict(
            obs_size = obs_size,
            gamma = 0.999,
            chi = 0.001,
            lr_actor = 0.05,
            n_hidden_act = num_hidden_a,
            hidden_size_act = hidden_size_a,
            decayRate = 0.999,
            alpha = 0.1, # introspection level
            c_value = args.c_value
            )
        
    if (args.algorithm == "actor-critic"):  
        num_hidden_a = 1
        hidden_size_a = 4
        obs_size = 2 # m factor and action of other
        algo_params = dict(
            obs_size = obs_size,
            gamma = 0.999,
            chi = 0.001,
            lr_actor = 0.05,
            n_hidden_act = num_hidden_a,
            hidden_size_act = hidden_size_a,
            decayRate = 0.999,
            alpha = 0.1, # introspection level
            c_value = args.c_value
            )

    elif (args.algorithm == "dqn"):
        obs_size = 2  # m factor and action of other
        algo_params = dict(
            obs_size = obs_size, # mult factor and reputation of opponent
            gamma = 0.99,
            chi = 0.001,
            epsilon = args.epsilon_dqn,
            memory_size = 500,
            n_hidden_act = 1,
            hidden_size_act = 4,
            dqn_activation_function = args.dqn_activation_function,
            lr_actor = args.lr_dqn,
            decayRate = 0.999,
            target_net_update_freq = 30,
            alpha = 0.1, # introspection level
            decaying_epsilon = True
        )

    elif (args.algorithm == "q-learning"):
        obs_size = 2  # m factor and action of other
        algo_params = dict(
            obs_size = obs_size,
            gamma = 0.99,
            chi = 0.001,
            epsilon = 0.01,
            lr_actor = 0.01,
            alpha = 0.1, # introspection level
        )

    elif (args.algorithm == "ppo"):
        obs_size = 2  # m factor and action of other
        num_steps = 10 #e` la stessa cosa di num_game_iterations
        num_minibatches = 4
        batch_size = num_steps #args.num_game_iterations
        total_timesteps = args.num_epochs
        minibatch_size = batch_size #int(batch_size // num_minibatches)
        algo_params = dict(
            obs_size = obs_size,
            gamma = 0.99,
            chi = 0.001,
            epsilon = 0.01,
            lr_actor = 0.001,
            anneal_lr = True,
            hidden_size_act = 4,
            num_steps = num_steps, ##128, #the number of steps to run in each environment per policy rollout
            gae_lambda = 0.95, #the lambda for the general advantage estimation
            num_minibatches = num_minibatches, #the number of mini-batches
            update_epochs = 4, #the K epochs to update the policy
            norm_adv = True, #Toggles advantages normalization
            clip_coef = 0.2, #the surrogate clipping coefficient
            clip_vloss = True, #Toggles whether or not to use a clipped loss for the value function, as per the paper.
            ent_coef = 0.01, #coefficient of the entropy
            vf_coef = 0.5, #coefficient of the value function
            max_grad_norm = 0.5, #the maximum norm for the gradient clipping
            target_kl = None, #the target KL divergence threshold,
            torch_deterministic = True, #if toggled, `torch.backends.cudnn.deterministic=False
            total_timesteps = total_timesteps,
            batch_size = batch_size, #the batch size (computed in runtime),
            num_iterations = total_timesteps // batch_size, # SAREBBE IL NUMERO DI EPOCHE
            minibatch_size = minibatch_size #the mini-batch size (computed in runtime)
            #alpha = 0.1, # introspection level
        )

    n_dummy = int(args.proportion_dummy_agents*args.n_agents)
    is_dummy = list(reversed([1 if i<n_dummy else 0 for i in range(args.n_agents) ]))
    non_dummy_idxs = [i for i,val in enumerate(is_dummy) if val==0]

    all_params = {**all_params, **game_params, **algo_params,  "n_dummy":n_dummy, "is_dummy":is_dummy, "non_dummy_idxs":non_dummy_idxs}

    return all_params