import numpy as np
import pandas as pd
from ramo.strategy.best_response import calc_expected_returns
import matplotlib as mpt
from ramo.game.monfg import MONFG
from ramo.nash.IBR import iterated_best_response
from ramo.nash.verify import verify_nash
from ramo.pareto.pareto_nash import pure_strategy_pne
import matplotlib.pyplot as plt

from utils import u_beta
from ramo.nash.verify import verify_nash

c = 4

dots = list(np.linspace(0, 1, 10)) + \
        list(np.linspace(0, 0.9, 20)) + \
        list(np.linspace(0.9, 1.1, 20)) + \
        list(np.linspace(1.1, 4.0, 40))
betas = dots
f_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpt.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

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

def find_nashes(_filename):

    print("\nFinding Nash equilibria and saving them to", _filename)

    data = []
    data_dict = {}; num_Nashes = {}; other = {}
    columns = ['f', 'beta', 'JointStrategy','JointStrategy[0]','JointStrategy[1]','SER1', 'SER2']

    for f in f_vals:

        payoffs = [np.array([[(c*f, 0), (c*f/2, 0)],
                             [(c*f/2, c), (0, c)]], dtype=float),
                   np.array([[(c*f, 0), (c*f/2, c)],
                             [(c*f/2, 0), (0, c)]], dtype=float)]
        
        game = MONFG(payoffs)

        for beta in betas:
            num_Nashes[(f,beta)] = 0

            u_tpl = (u_beta([1, 1], beta), u_beta([1, 1], beta))

            print('\nRunning for f =', f, 'and beta =', beta)

            init_strats = [np.array([[1., 0.], [1., 0.]]), \
                           np.array([[1., 0.], [0., 1.]]), \
                           np.array([[0., 1.], [0., 1.]]), \
                           np.array([[0.5, 0.5], [0.5, 0.5]])]
            
            for init_strat in init_strats:
                is_ne, joint_strat = iterated_best_response(game, u_tpl, global_opt=True,
                                                            init_joint_strategy=init_strat, max_iter=10000)
                js = np.array((joint_strat[0][0],joint_strat[0][1],joint_strat[1][0],joint_strat[1][1]))

                if (is_ne):
                    ser = get_ser(game, joint_strat, u_tpl)
                    if (f, beta) not in data_dict:
                        num_Nashes[(f,beta)] = 1
                        data_dict[(f,beta)] = [joint_strat]
                        other[(f,beta)] = [js]
                        data.append([f, beta, joint_strat, 
                                    joint_strat[0][0], joint_strat[1][0], # saving only cooperation
                                    ser[0], ser[1]])
                    else:
                        new_element_to_add = []
                        for element in other[(f,beta)]:
                            if (abs(js - element)>10E-15).all():
                                new_element_to_add.append(True)
                            else: 
                                new_element_to_add.append(False)
                        if (new_element_to_add == [True for i in range(len(new_element_to_add))]):
                            num_Nashes[(f,beta)] += 1
                            data_dict[(f,beta)].append(joint_strat)
                            other[(f,beta)].append(js)
                            data.append([f, beta, joint_strat, 
                                        joint_strat[0][0], joint_strat[1][0], # saving only cooperation
                                        ser[0], ser[1]])

            # Check if the PNE are SER Nash equilibrium
            pne_strats = pure_strategy_pne(game)
            for el in pne_strats:
                vNE = verify_nash(game, u_tpl, el)
                js = np.array((el[0][0],el[0][1],el[1][0],el[1][1]))
                if (vNE):
                    if (f, beta) not in data_dict:
                        num_Nashes[(f,beta)] = 1
                        data_dict[(f,beta)] = [joint_strat]
                        other[(f,beta)] = [js]
                        data.append([f, beta, joint_strat, 
                                    joint_strat[0][0], joint_strat[1][0], # saving only cooperation
                                    ser[0], ser[1]])
                    else:
                        new_element_to_add = []
                        for element in other[(f,beta)]:

                            if (abs(js - element)>10E-15).all():
                                new_element_to_add.append(True)
                            else: 
                                new_element_to_add.append(False)
                        if (new_element_to_add == [True for i in range(len(new_element_to_add))]):
                            num_Nashes[(f,beta)] += 1
                            data_dict[(f,beta)].append(el)
                            other[(f,beta)].append(js)
                            data.append([f, beta, js, 
                                        el[0][0], el[1][0], # saving only cooperation
                                        ser[0], ser[1]])

                    ser = get_ser(game, el, u_tpl)
            print("num Nashes found for (f=", f, ", beta=", beta, ") =", num_Nashes[(f,beta)])
    
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(_filename, index=False)
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(_filename, index=False)
    print("Saved all found data in", _filename)

def plots_probability_of_cooperation(_file):

    df = pd.read_csv(_file)
    f_vals = df["f"].unique()
    f_vals = [0.5, 1.0, 1.5, 2., 2.5, 3.]
    betas = df['beta'].unique()

    print("Making plots for Probability of Cooperation for f values:", f_vals, "and betas from", min(betas), "to", max(betas))
    print("Loading data from", _file)

    new_cmap = truncate_colormap(plt.cm.CMRmap, 0.0, 0.8)
    colors = new_cmap(np.linspace(0,1,len(f_vals)))
    values_anchor = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

    for i in range(2):
        fig, ax = plt.subplots(len(f_vals), 1, figsize=(8,13))
        plt.subplots_adjust(wspace=0, hspace=0.05)
        plt.title("Nash Agent "+str(i), fontsize=25)

        for _if, f in enumerate(f_vals):
            data_f = df.loc[df['f'] == f]
            ax[_if].scatter(data_f['beta'], data_f['JointStrategy['+str(i)+']'], label="f value:"+str(f), color=colors[_if], s=25)#, marker=markers[_if])

            handle_list, label_list = [], []
            handles, labels = ax[_if].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_list.append(handle)
                    label_list.append(label)

            plt.rcParams["legend.title_fontsize"] = 15

            for lbl in ax[_if].get_yticklabels():
                lbl.set_fontsize(15)
            for lbl in ax[_if].get_xticklabels():
                lbl.set_fontsize(15)

            ax[_if].grid(alpha=0.5)
            ax[_if].legend(fontsize='x-large',bbox_to_anchor=(1, values_anchor[_if]),ncol=len(df.columns))

        ax[_if].set_xlabel(r'$\beta$', fontsize=20)
        fig.text(0, 0.5, 'Probability of Cooperation', fontsize=18, va='center', rotation='vertical')
        filename = "poc_agent"+str(i)+".png"
        fig.savefig("results/"+filename, bbox_inches="tight")
        print("Saved plot for agent", i, "in results/"+filename)

def plots_poa(_file):

    df = pd.read_csv(_file)
    f_vals = df["f"].unique()  
    betas = df['beta'].unique()
    print("\nPlotting the Price of Anarchy for f values:", f_vals, "and betas from", min(betas), "to", max(betas))
    print("Loading data from", _file)

    f_vals = [0.5, 1.0, 1.5, 2., 2.5, 3.]
    new_cmap = truncate_colormap(plt.cm.CMRmap, 0.0, 0.8)
    colors = new_cmap(np.linspace(0,1,len(f_vals)))
    poa_array = np.zeros((len(f_vals), len(betas)))
    poa_f_s = []

    res_strategy = 0.1
    X1 = np.linspace(0.,1.,int(np.rint((1.+res_strategy)/res_strategy)))
    Y1 = np.linspace(0.,1.,int(np.rint((1.+res_strategy)/res_strategy)))
    ser_agent0 = np.zeros((len(X1), len(Y1)))
    fig, ax = plt.subplots(1, 1, figsize=(7,5))

    for _if, f in enumerate(f_vals):
        poa_f = []
        payoffs = [np.array([[(c*f, 0), (c*f/2, 0)],
                             [(c*f/2, c), (0, c)]], dtype=float),
                   np.array([[(c*f, 0), (c*f/2, c)],
                             [(c*f/2, 0), (0, c)]], dtype=float)]
        
        game = MONFG(payoffs)

        data_f = df.loc[df['f'] == f]
        poa_old = 0

        for ib, beta in enumerate(betas):

            df_f_beta = data_f.loc[data_f['beta'] == beta]
            u_tpl = (u_beta([1, 1], beta), u_beta([1, 1], beta))

            max_welfare = -10000
            min_welfare_nash = 100000

            welfares = np.zeros((len(X1), len(Y1)))

            for i in range(len(X1)):
                for j in range(len(Y1)):
                    strat = np.array([[X1[i], 1.-X1[i]], [Y1[j], 1.-Y1[j]]])
                    sers = get_ser(game, strat, u_tpl)
                    welfare = np.sum(sers)
                    if welfare > max_welfare:
                        max_welfare = welfare
                    welfares[i,j] = welfare
                    ser_agent0[i,j] = sers[0]

            for i in range(len(df_f_beta)):
                # pick the nash strategies and find the min welfare one
                pc0 = df_f_beta["JointStrategy[0]"].tolist()[i]
                pc1 = df_f_beta["JointStrategy[1]"].tolist()[i]
                strat_nash = np.array([[pc0, 1.-pc0], [pc1, 1.-pc1]])

                sers = get_ser(game, strat_nash, u_tpl)
                welfare_nash = np.sum(sers)
    
                if welfare_nash < min_welfare_nash:
                    min_welfare_nash = welfare_nash

            poa = max_welfare/min_welfare_nash
            if (f == 1.5):
                if (poa < 1.):
                    poa = poa_old
                else: 
                    poa_old = poa
            poa_f.append(poa)
        
        poa_f_s.append(poa_f)
        ax.plot(betas, poa_f, label=f, color=colors[_if])
        poa_array[_if, ib] = poa
        plt.rcParams["legend.title_fontsize"] = 15

        ax.grid(alpha=0.5)
        ax.set_xlim(0,4)
        ax.legend(title="f values",fontsize='x-large',bbox_to_anchor=(1.0, 0.8))#,ncol=len(df.columns))
    ax.set_xlabel(r'$\beta$', fontsize=19)
    ax.set_ylabel("Price of Anarchy", fontsize=19)
    fig.savefig("results/poa.png", bbox_inches="tight")

    print("Saved plot for the Price of Anarchy in results/poa.png")

if __name__ == "__main__":
    
    # Run once to get the data file, then comment it out and run the plotting functions
    #find_nashes("results/data.csv")

    # Make plot for the Probability of Cooperation for each agent, given a set of f values and betas (from file)
    plots_probability_of_cooperation("results/data.csv")

    # Make plot for the Price of Anarchy of the system, given a set of f values and betas (from file)
    plots_poa("results/data.csv")