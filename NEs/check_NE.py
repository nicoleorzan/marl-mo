import math

import numpy as np
import itertools
import pandas as pd
from ramo.strategy.best_response import calc_expected_returns
import decimal
from utils import mkdir_p
from ramo.game.monfg import MONFG
from ramo.nash.fictitious_play import fictitious_play
from ramo.nash.IBR import iterated_best_response
from ramo.nash.verify import verify_nash
from ramo.pareto.pareto_nash import pure_strategy_pne
import matplotlib.pyplot as plt

from utils import ROW_PLAYER, COL_PLAYER, u_beta, pgg_f_05, pgg_f_15, pgg_f_25, generate_strategies, mkdir_p
from ramo.nash.verify import verify_nash

def get_ser(game, joint_strat, u_tpl):
    ser = np.zeros(2)
    for player, (payoff_matrix, strat) in enumerate(zip(game.payoffs, joint_strat)):
        expected_returns = calc_expected_returns(player, payoff_matrix, joint_strat)
        # Expected vector under considered strategy
        expected_vec = np.dot(strat, expected_returns)
        ser[player] = u_tpl[player](expected_vec)
    return ser


def is_NE(game, joint_strat, u_tpl):
    vNE = verify_nash(game, u_tpl, joint_strat)
    print(vNE)
    ser = get_ser(game, joint_strat, u_tpl)
    print(joint_strat, ser[0], ser[1])


if __name__ == "__main__":
    beta = 0.5
    f = 1.5

    u_tpl = (u_beta([1, 1], beta), u_beta([1, 1], beta))

    print('Running for f =', f, 'and beta =', beta)
    if f == 0.5:
        game = pgg_f_05
    elif f == 1.5:
        game = pgg_f_15
    elif f == 2.5:
        game = pgg_f_25

    # init_strat = np.array([[1., 0.], [0., 1.]])
    # init_strat = np.array([[0.5, 0.5], [0.5, 0.5]])
    # init_strat = np.array([[0., 1.], [0., 1.]])
    init_strat = np.array([[1., 0.], [1., 0.]])
    is_ne, joint_strat = iterated_best_response(game, u_tpl, global_opt=True,
                                                init_joint_strategy=init_strat, max_iter=10000)
    print("IBR result:", is_ne, joint_strat)
    is_NE(game, joint_strat, u_tpl)

    # epsilon = 1e-10
    # joint_strat[0][0] += epsilon
    # joint_strat[0][1] -= epsilon
    # is_NE(game, joint_strat, u_tpl)

    # is_ne, joint_strat, log = fictitious_play(game, u_tpl, global_opt=False)
    # print(is_ne, joint_strat)

    # Check if the PNE are SER Nash equilibrium
    pne_strats = pure_strategy_pne(game)
    for el in pne_strats:
        vNE = verify_nash(game, u_tpl, el)
        print(vNE)
        ser = get_ser(game, el, u_tpl)
        print(el, ser[0], ser[1])

