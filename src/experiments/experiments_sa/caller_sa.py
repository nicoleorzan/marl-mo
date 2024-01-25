import argparse
import numpy as np
from src.experiments.experiments_sa.train_dqn_sa import train_dqn

def non_increasing(L):
    return all(x>y for x, y in zip(L, L[1:]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int)
    parser.add_argument(
        "--mult_fact",
        nargs="*",
        type=float,
        default=[1.5])
    parser.add_argument( # to fill with values of uncertainties for every agent (can be 0.)
        "--uncertainties",
        nargs="*",
        type=float,
        default=[])
        
    parser.add_argument('--optuna_', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_game_iterations', type=int, default=1)
    parser.add_argument('--num_objectives', type=int, choices = [1, 2, 3], default=1)
    parser.add_argument('--n_active_agents', type=int, default=2)
    parser.add_argument('--reputation_enabled', type=int, default=0)
    parser.add_argument('--coins_value', type=float, default=4.)
    parser.add_argument('--proportion_dummy_agents', type=float, default=0.)
    parser.add_argument('--binary_reputation', type=int, choices = [0, 1], default=1) # 1 yes 0 no
    parser.add_argument('--wandb_mode', type=str, choices = ["online", "offline"], default="offline")
    parser.add_argument('--dqn_activation_function', type=str, choices = ["tanh", "relu"], default="tanh")
    parser.add_argument('--rule', type=str, choices = ["rule09", "rule11", "rule03", "rule00"], default="rule09")
    parser.add_argument('--optimize', type=int, choices = [0, 1], default=0) # 1 for true 0 for false
    parser.add_argument('--introspective', type=int, choices = [0, 1], default=0) # 1 for true 0 for false
    parser.add_argument('--reputation_assignment', type=int, choices = [0, 1], default=0) # 1 for true 0 for false
    parser.add_argument('--mf_from_interval', type=int, choices = [0, 1], default=0) # 1 for true 0 for false
    parser.add_argument('--scalarization_function', type=str, choices = ["linear", "ggf", "g3f"], default="linear")
    parser.add_argument('--print_step', type=int, default=50)
    parser.add_argument('--epsilon_dqn', type=float, default=0.01)
    parser.add_argument('--lr_dqn', type=float, default=0.001)
    parser.add_argument( # to fill with values of weights for every objective (can be 0.)
        "--weights",
        nargs="*",
        type=float,
        default=[])
    
    parser.add_argument( # to fill with values of weights for every objective (can be 0.)
        "--weights_agents",
        nargs="*",
        type=float,
        default=[])
    
    parser.add_argument( # to fill with values of weights for every objective (can be 0.)
        "--agents_funcs",
        nargs="*",
        type=str,
        default=['linear'])

    args = parser.parse_args()
    n_certain_agents = args.uncertainties.count(0.)
    n_uncertain = args.n_agents - n_certain_agents
    assert(args.n_active_agents <= args.n_agents)

    # if GGF is employed
    if (args.scalarization_function == "ggf"):
        #weights should sum to 1
        assert(np.abs(np.sum(args.weights) - 1.0) < 10E-6)
        #weights should be organized in descending order:  w_1 > ... > w_n
        assert (non_increasing(args.weights) == True) 
    #print("np.sum(args.weights)=",np.sum(args.weights))

    if (args.scalarization_function == "g3f"):
        assert (non_increasing(args.weights) == True) 
        assert(np.abs(np.sum(args.weights) - 1.0) < 10E-6)
        assert(np.abs(np.sum(args.p_weights) - 1.0) < 10E-6)
    
    assert(args.proportion_dummy_agents >= 0.)    
    assert(args.proportion_dummy_agents <= 1.)

    assert(args.n_agents > 1)
    assert(len(args.uncertainties) == args.n_agents)
    assert(len(args.weights) == args.n_active_agents)
    assert(len(args.weights_agents) == args.num_objectives)

    if (args.reputation_enabled == 0):
        assert(args.proportion_dummy_agents == 0)

    train_dqn(args)