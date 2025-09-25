import numpy as np
from utils import ROW_PLAYER, COL_PLAYER, u_beta, pgg_f_05, pgg_f_15, pgg_f_25, generate_strategies, mkdir_p
from ramo.nash.verify import verify_nash
from ramo.strategy.best_response import calc_expected_returns
import pandas as pd
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, default='results', help="Folder to save data in.")
    parser.add_argument('-f', type=float, default=2.5, help="The f value of the game.")
    parser.add_argument('-res_strategy', type=float, default=0.001, help="Strategy resolution")
    parser.add_argument('-beta', type=float, default=1., help="Beta value for the utility function")

    args = parser.parse_args()

    folder = args.folder
    mkdir_p(folder)

    f = args.f
    res_strategy = args.res_strategy
    beta = args.beta

    print('Finding Nash equilibria for a two-player Extended Public Goods Game')
    print('parameters: f =', f, ', beta =', beta, ', strategy resolution:', res_strategy)
    print('Might take a while... (consider increasing the strategy resolution if it takes too long)')

    if f == 0.5:
        game = pgg_f_05
    elif f == 1.5:
        game = pgg_f_15
    elif f == 2.5:
        game = pgg_f_25

    # Generate a list with all possible individual strategies at the specified resolution
    all_joint_strategies = generate_strategies(res_strategy)
    data = [[], []]
    columns = ['O1', 'O2', 'JointStrategy', 'SER']
    utility_tuple = (u_beta([1, 1], beta), u_beta([1, 1], beta))
    #print("utility_tuple=", utility_tuple)
    # Identify the joint strategies that are Nash equilibria
    expected_payoffs = [[], []]
    for joint_strat in all_joint_strategies:
        #print("joint strategy=", joint_strat)
        #utility = 1 * (np.abs(p[0]) ** beta) + (1 * p[1])
        is_ne = verify_nash(game, utility_tuple, joint_strat)
        if is_ne:
            for player, (payoff_matrix, strat) in enumerate(zip(game.payoffs, joint_strat)):
                expected_returns = calc_expected_returns(player, payoff_matrix, joint_strat)
                # Expected vector under considered strategy
                expected_vec = np.dot(strat, expected_returns)
                ser = utility_tuple[player](expected_vec)
                data[player].append([np.around(expected_vec[0], decimals=5),
                                     np.around(expected_vec[1], decimals=5),
                                     np.around(joint_strat, decimals=5),
                                     round(ser, 5)])

    for player in [ROW_PLAYER, COL_PLAYER]:
        file = f'{folder}/pgg_f{f}_beta{beta}_NEs_res_s{res_strategy}_player{player}.csv'
        df = pd.DataFrame(data[player], columns=columns)
        df.to_csv(file, index=False)
        print(f'Saved data for player {player} in {file}.')
