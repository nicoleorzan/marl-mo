import torch

ACTION_SIZE = 2
RANDOM_BASELINE = False

def setup_training_hyperparams(args, trial):

    all_params = {}

    game_params = dict(
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu"),
        num_epochs = args.num_epochs,
        seed = 123,
        action_size = ACTION_SIZE,
        print_step = 50,
        wandb_mode = args.wandb_mode
    )

    obs_size = 2  # m factor and action of other
    num_minibatches = 4
    batch_size = args.num_steps #args.num_game_iterations
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
        num_steps = args.num_steps, ##128, #the number of steps to run in each environment per policy rollout
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
    )

    all_params = {**all_params, **game_params, **algo_params}

    return all_params