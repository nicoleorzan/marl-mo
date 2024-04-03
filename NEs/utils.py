import os
import errno
import numpy as np
from ramo.game.monfg import MONFG

DEF = 0
COOP = 1
ROW_PLAYER = 0  # DO NOT CHANGE - variable with player index
COL_PLAYER = 1  # DO NOT CHANGE - variable with player index
N = 1000

ACTIONS_2 = [DEF, COOP]
NUM_OBJ = 2


def df2array(df):
    return df.apply(lambda x: np.fromstring(
        x.replace('\n',
                  '')
        .replace(
            '[', '')
        .replace(
            ']', '')
        .replace(
            '  ', ' '),
        sep=' '))


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def u_beta(w, beta):
    # Had to add np.abs to avoid errors due to potential negative values when beta < 1, but p for us is always > 0
    #print("(np.abs(p[0]) ** beta) =", (np.abs(p[0]) ** beta) )
    return lambda p: w[0] * (np.abs(p[0]) ** beta) + (w[1] * p[1])


payoffs05 = [np.array([[(2, 0), (1, 0)],
                       [(1, 4), (0, 4)]], dtype=float),
             np.array([[(2, 0), (1, 4)],
                       [(1, 0), (0, 4)]], dtype=float)]

payoffs15 = [np.array([[(6, 0), (3, 0)],
                       [(3, 4), (0, 4)]], dtype=float),
             np.array([[(6, 0), (3, 4)],
                       [(3, 0), (0, 4)]], dtype=float)]

payoffs25 = [np.array([[(10, 0), (5, 0)],
                       [(5, 4), (0, 4)]], dtype=float),
             np.array([[(10, 0), (5, 4)],
                       [(5, 0), (0, 4)]], dtype=float)]

pgg_f_05 = MONFG(payoffs05)
pgg_f_15 = MONFG(payoffs15)
pgg_f_25 = MONFG(payoffs25)

def generate_all_individual_strategies(resolution):
    # Generate a list with all possible individual strategies at the specified resolution
    # This only works for 2 actions right now, need to reimplement to make it work for any number of actions
    initial = np.array([1.0, 0.0])
    all_individual_strategies = [initial.copy()]
    while initial[0] > 0.00001:  # use a small number close to zero to avoid errors due to precision
        initial[0] = initial[0] - resolution
        initial[1] = initial[1] + resolution
        all_individual_strategies.append(initial.copy())
    return all_individual_strategies


def generate_strategies(resolution):
    individual_strategies = generate_all_individual_strategies(resolution)
    all_joint_strategies = []
    for i in range(len(individual_strategies)):
        for j in range(len(individual_strategies)):
            all_joint_strategies.append([individual_strategies[i].copy(), individual_strategies[j].copy()])
    return all_joint_strategies
