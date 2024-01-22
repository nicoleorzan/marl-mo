import argparse
import numpy as np
from src.experiments.train_reinforce import train_reinforce
from src.experiments.train_q_learning import train_q_learning
from src.experiments.train_dqn import train_dqn

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
    parser.add_argument('--reputation_enabled', type=int, default=0)
    parser.add_argument('--coins_value', type=float, default=4.)
    parser.add_argument('--proportion_dummy_agents', type=float, default=0.)
    parser.add_argument('--binary_reputation', type=int, choices = [0, 1], default=1) # 1 yes 0 no
    parser.add_argument('--wandb_mode', type=str, choices = ["online", "offline"], default="offline")
    parser.add_argument('--dqn_activation_function', type=str, choices = ["tanh", "relu"], default="tanh")
    parser.add_argument('--rule', type=str, choices = ["rule09", "rule11", "rule03", "rule00"], default="rule09")
    parser.add_argument('--algorithm', type=str, choices = ["reinforce", "PPO", "dqn", "q-learning"], default="reinforce")
    parser.add_argument('--optimize', type=int, choices = [0, 1], default=0) # 1 for true 0 for false
    parser.add_argument('--introspective', type=int, choices = [0, 1], default=0) # 1 for true 0 for false
    parser.add_argument('--reputation_assignment', type=int, choices = [0, 1], default=0) # 1 for true 0 for false
    parser.add_argument('--mf_from_interval', type=int, choices = [0, 1], default=0) # 1 for true 0 for false
    parser.add_argument('--scalarization_function', type=str, choices = ["linear", "ggf"], default="linear")
    parser.add_argument('--print_step', type=int, default=50)
    parser.add_argument('--epsilon_dqn', type=float, default=0.01)
    #parser.add_argument('--_print', type=int, choices = [0, 1], default=0) # 1 yes 0 no
    parser.add_argument( # to fill with values of weights for every objective (can be 0.)
        "--weights",
        nargs="*",
        type=float,
        default=[])

    args = parser.parse_args()
    n_certain_agents = args.uncertainties.count(0.)
    n_uncertain = args.n_agents - n_certain_agents

    # if GGF is employed
    #check if weights are organized in descending order:  w_1 > ... > w_n
    #print("args.weights",args.weights)
    #print("non inc?", non_increasing(args.weights))
    if (args.scalarization_function == "ggf"):
        assert (non_increasing(args.weights) == True) 
    print("np.sum(args.weights)=",np.sum(args.weights))
    assert(np.abs(np.sum(args.weights) - 1.0) < 10E-6)
    #w = torch.Tensor(args.weights)
    #print("w=", w)
    #assert(torch.sum(w) == 1)
    #if (args.scalarization_function == "ggf"):
    #    assert(w.sort(descending=True)[0] == w).all()
    
    assert(args.proportion_dummy_agents >= 0.)    
    assert(args.proportion_dummy_agents <= 1.)

    assert(args.n_agents > 1)
    if (args.algorithm == "q-learning"):
        assert(n_certain_agents == args.n_agents)
    assert(len(args.uncertainties) == args.n_agents)
    assert(len(args.weights) == args.num_objectives)

    if (args.reputation_enabled == 0):
        assert(args.proportion_dummy_agents == 0)

    if args.algorithm == "dqn":
        train_dqn(args)
    elif args.algorithm == "reinforce":
        print("calling reinforce")
        train_reinforce(args)    
    elif args.algorithm == "q-learning":
        train_q_learning(args)
