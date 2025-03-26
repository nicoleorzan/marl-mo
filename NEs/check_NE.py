import math

import numpy as np
import itertools
import pandas as pd
from ramo.strategy.best_response import calc_expected_returns
import decimal
import seaborn as sns
from utils import mkdir_p
import matplotlib as mpt
from matplotlib.colors import LogNorm
from ramo.game.monfg import MONFG
from ramo.nash.fictitious_play import fictitious_play
from ramo.nash.IBR import iterated_best_response
from ramo.nash.verify import verify_nash
from ramo.pareto.pareto_nash import pure_strategy_pne
import matplotlib.pyplot as plt

from utils import ROW_PLAYER, COL_PLAYER, u_beta, pgg_f_05, pgg_f_15, pgg_f_25, generate_strategies, mkdir_p
from ramo.nash.verify import verify_nash

c = 4
#file = 'data/pgg_NEs_new.csv'
betas = np.linspace(1, 2, 10)
dots = list(np.linspace(-1, 1, 20)) + \
        list(np.linspace(0, 0.9, 20)) + \
        list(np.linspace(0.9, 1.1, 20)) + \
        list(np.linspace(1.1, 4.0, 40))
#dots =  list(np.linspace(0.0, 4.0, 20))
betas = dots
#betas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
f_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]# 5.0] 5.5, 6.0, 6.5]
#f_vals = [3.0, 3.5, 4.0, 4.5, 5.0]

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
                #print("searching Nash from ",init_strat.reshape(-1))
                is_ne, joint_strat = iterated_best_response(game, u_tpl, global_opt=True,
                                                            init_joint_strategy=init_strat, max_iter=10000)
                #print("IBR result:", is_ne, joint_strat)
                #print(joint_strat)
                js = np.array((joint_strat[0][0],joint_strat[0][1],joint_strat[1][0],joint_strat[1][1]))

                if (is_ne):
                    ser = get_ser(game, joint_strat, u_tpl)
                    if (f, beta) not in data_dict:
                        print("a found Nash:", joint_strat)
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
                            print("b adding Nash:", joint_strat)
                            num_Nashes[(f,beta)] += 1
                            data_dict[(f,beta)].append(joint_strat)
                            other[(f,beta)].append(js)
                            data.append([f, beta, joint_strat, 
                                        joint_strat[0][0], joint_strat[1][0], # saving only cooperation
                                        ser[0], ser[1]])
                                        
                #print("is_ne:")
                #is_NE(game, joint_strat, u_tpl)

            # epsilon = 1e-10
            # joint_strat[0][0] += epsilon
            # joint_strat[0][1] -= epsilon
            # is_NE(game, joint_strat, u_tpl)

            # is_ne, joint_strat, log = fictitious_play(game, u_tpl, global_opt=False)
            # print(is_ne, joint_strat)

            # Check if the PNE are SER Nash equilibrium
            print("Checking pne")
            pne_strats = pure_strategy_pne(game)
            for el in pne_strats:
                #print("Checking for=", el)
                vNE = verify_nash(game, u_tpl, el)
                js = np.array((el[0][0],el[0][1],el[1][0],el[1][1]))
                if (vNE):
                    if (f, beta) not in data_dict:
                        num_Nashes[(f,beta)] = 1
                        print("c found Nash:", joint_strat)
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
                            print("b adding Nash:", js)
                            num_Nashes[(f,beta)] += 1
                            data_dict[(f,beta)].append(el)
                            other[(f,beta)].append(js)
                            data.append([f, beta, js, 
                                        el[0][0], el[1][0], # saving only cooperation
                                        ser[0], ser[1]])

                    ser = get_ser(game, el, u_tpl)
            print("num Nashes for", (f, beta), "=", num_Nashes[(f,beta)])
    
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(_filename, index=False)
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(_filename, index=False)

def strategy_plot(_filename):

    df = pd.read_csv(_filename)
    betas = df['beta'].unique()
    f_vals = df['f'].unique()

    colors = plt.cm.jet(np.linspace(0,1,len(betas)))

    for _if, f in enumerate(f_vals):
        print("if=", _if)
        data_f = df.loc[df['f'] == f]
        #print("\ndata_f",len(data_f.index))

        fig, ax = plt.subplots(1)

        for ib, beta in enumerate(betas):
            #print("beta=",data_f['beta'])
            #print("js0=", data_f['JointStrategy[0]'])
            #print("js1=", data_f['JointStrategy[1]'])
            #if (beta == betas[0])
            ax.scatter(data_f['JointStrategy[0]'], data_f['JointStrategy[1]'], label=str(beta), color=colors[ib], s=10)

        """
        handles, labels = ax.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        plt.legend(handle_list, label_list, title=r"$\beta$ value", fontsize=13)
        """

        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('Probability of Cooperation agent 1', fontsize=13)
        plt.ylabel('Probability of Cooperation agent 0', fontsize=13)
        plt.savefig('new_strategy_space_f'+str(f)+'.png')

def make_plots(_file):

    df = pd.read_csv(_file)
    f_vals = df["f"].unique()
    f_vals = [0.5, 1.0, 1.5, 2., 2.5, 3.]
    #markers = [".", "s", "x", "D", "P", "p"]
    betas = df['beta'].unique()
    print("betas=", betas)

    #colors = plt.cm.jet(np.linspace(0,1,len(f_vals)))
    new_cmap = truncate_colormap(plt.cm.CMRmap, 0.0, 0.8)
    colors = new_cmap(np.linspace(0,1,len(f_vals)))
    values_anchor = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    #handle_list, label_list = [], []

    for i in range(2):
        fig, ax = plt.subplots(len(f_vals), 1, figsize=(8,13))#, gridspec_kw={"wspace":0.01})
        plt.subplots_adjust(wspace=0, hspace=0.05)
        #plt.title("Nash Agent "+str(i), fontsize=25)
        print("f_vals=", f_vals)

        for _if, f in enumerate(f_vals):
            print("if=", _if, "f=", f)
            data_f = df.loc[df['f'] == f]
            print("len=",len(data_f.index))
            #for beta in betas:
            print("beta=",list(data_f['beta'])[0:3])
            print(list(data_f['JointStrategy[0]'])[0:3])
            #print("f=", data_f['JointStrategy[1]'])
            #ax.plot(data_f['beta'], data_f['JointStrategy['+str(i)+']'], label=str(f), color=colors[_if])#, s=16)
            ax[_if].scatter(data_f['beta'], data_f['JointStrategy['+str(i)+']'], label="f value:"+str(f), color=colors[_if], s=25)#, marker=markers[_if])

            handle_list, label_list = [], []
            handles, labels = ax[_if].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_list.append(handle)
                    label_list.append(label)

            plt.rcParams["legend.title_fontsize"] = 15
            for tick in ax[_if].yaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            for tick in ax[_if].xaxis.get_major_ticks():
                tick.label.set_fontsize(0)

            ax[_if].grid(alpha=0.5)
            ax[_if].legend(fontsize='x-large',bbox_to_anchor=(1, values_anchor[_if]),ncol=len(df.columns))
            #ax[_if].set_xlim(0,3)

        for tick in ax[_if].xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        ax[_if].set_xlabel(r'$\beta$', fontsize=20)
        #ax[_if].set_ylabel('Probability of Cooperation', fontsize=20)
        #plt.legend(handle_list, label_list, title="f value", fontsize='x-large', framealpha=0.3, bbox_to_anchor=(1.1, 6.8),ncol=len(df.columns))
        fig.text(0, 0.5, 'Probability of Cooperation', fontsize=18, va='center', rotation='vertical')
        fig.savefig("nuovo_agent"+str(i)+".png", bbox_inches="tight")

def poa(_file):

    df = pd.read_csv(_file)
    f_vals = df["f"].unique()  
    betas = df['beta'].unique()

    """betas = []
    beta_chosen = 0.8
    for bee in betas1:
        #print("bee=", bee)
        if abs(bee - beta_chosen)<10e-3:
            betas.append(bee)
            break

    print("betas=", betas)"""

    #betas = [0.5, 1., 1.5, 2., 2.5, 3]#, 4.5, 5]
    f_vals = [0.5, 1.0, 1.5, 2., 2.5, 3.]
    new_cmap = truncate_colormap(plt.cm.CMRmap, 0.0, 0.8)
    colors = new_cmap(np.linspace(0,1,len(f_vals)))
    print("f_vals=", f_vals)
    #betas = [2.5]
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
            print("\nf=", f, "beta=", beta)

            df_f_beta = data_f.loc[data_f['beta'] == beta]
            #print('df_f_beta=',df_f_beta)
            u_tpl = (u_beta([1, 1], beta), u_beta([1, 1], beta))

            max_welfare = -10000
            min_welfare_nash = 100000

            welfares = np.zeros((len(X1), len(Y1)))

            for i in range(len(X1)):
                for j in range(len(Y1)):
                    strat = np.array([[X1[i], 1.-X1[i]], [Y1[j], 1.-Y1[j]]])
                    #print("strat=",strat)
                    sers = get_ser(game, strat, u_tpl)
                    #print("sers=",sers)
                    welfare = np.sum(sers)
                    if welfare > max_welfare:
                        max_welfare = welfare
                        max_welfare_strat = strat
                    welfares[i,j] = welfare
                    ser_agent0[i,j] = sers[0]

            #print("\nFinding min Nash welfare")
            #print("df_f_beta=",df_f_beta)
            for i in range(len(df_f_beta)):
                # pick the nash strategies and find the min welfare one
                #print('df_f_beta["JointStrategy[0]"].tolist()=',df_f_beta["JointStrategy[0]"])
                #print("df_f_beta[JointStrategy[0]=",df_f_beta["JointStrategy[0]"])
                pc0 = df_f_beta["JointStrategy[0]"].tolist()[i]
                pc1 = df_f_beta["JointStrategy[1]"].tolist()[i]
                strat_nash = np.array([[pc0, 1.-pc0], [pc1, 1.-pc1]])
                #print("strat=",strat_nash)

                sers = get_ser(game, strat_nash, u_tpl)
                #print("sers=", sers)
                welfare_nash = np.sum(sers)
    
                #print("welfare_nash=",welfare_nash)
                if welfare_nash < min_welfare_nash:
                    min_welfare_nash = welfare_nash
                    min_welfare_strat_nash = strat_nash

            poa = max_welfare/min_welfare_nash
            if (f == 1.5):
                if (poa < 1.):
                    poa = poa_old
                else: 
                    poa_old = poa
            poa_f.append(poa)
            #print("\nmax_welfare=",max_welfare)
            ##print("max_welfare_strat=",max_welfare_strat)
            #print("min_welfare_nash",min_welfare_nash)
            #print("min_welfare_strat_nash=",min_welfare_strat_nash)

            print("poa=", poa)
            #print("sers_agent0=", ser_agent0)
            #sns.heatmap(ser_agent0)#, norm=LogNorm())      
            #plt.show()  
            sns.heatmap(welfares)#, norm=LogNorm())  
            plt.suptitle("f=" +str(f) + ", beta" + str(beta))
            plt.show()
            sns.heatmap(ser_agent0)#, norm=LogNorm())  
            plt.suptitle("ser agent 0")
            plt.show()

        """
        poa_f_s.append(poa_f)
        ax.plot(betas, poa_f, label=f, color=colors[_if])
        poa_array[_if, ib] = poa
        plt.rcParams["legend.title_fontsize"] = 15
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)

        ax.grid(alpha=0.5)
        ax.set_xlim(0,4)
        ax.legend(title="f values",fontsize='x-large',bbox_to_anchor=(1.0, 0.8))#,ncol=len(df.columns))
    ax.set_xlabel(r'$\beta$', fontsize=19)
    ax.set_ylabel("Price of Anarchy", fontsize=19)
    fig.savefig("poa.png", bbox_inches="tight")"""

    """for i in range(len(X1)):
        for j in range(len(Y1)):
            exp_vecs[:,i,j], new_sers[i,j], br_strat[player,i,j], is_nash[i,j] = compute_ser_nash_br(player, X1[i], Y1[j], global_opt=global_opt)
            if (player == 0):
                if (is_nash[i,j] == 1.):
                    nash_welfares[(i,j)] = new_sers[i,j]
                welfares[(i,j)] = new_sers[i,j]

            if (player == 1):
                welfares[(j,i)] += new_sers[i,j] # I save welfares based on the strategy of first player
                if (is_nash[i,j] == 1.):
                    nash_welfares[(j,i)] += new_sers[i,j]

    max_welfare = max(welfares.values())
    if (nash_welfares != {}):
        min_nash_welfare = min(nash_welfares.values())
        print("max_welfare=",max_welfare)
        print("min_nash_welfare=",min_nash_welfare)
        poa = max_welfare/min_nash_welfare
        print("Price of Anarchy=", poa)
    else: 
        print("No Nash to compute PoA")"""

if __name__ == "__main__":
    #print("betas=", betas)
    #print("f=",f_vals)
    
    ##print("data/for_strategy_plot"+str(f_vals)+"_new.csv")
    find_nashes("data/negative_beta.csv")

    #make_plots("data/hope_last.csv") #/new.csv")

    #strategy_plot("data/new.csv")

    #poa("data/hope_last.csv")
    #poa("data/for_strategy_plot[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]_new.csv")

    """c = 4.
    #f = 1./c + 0.00000001
    f = 1.5
    beta = math.log(c)/math.log(c*f) + 0.1
    print("c=", c, "f=", f, "beta=", beta)
    payoffs = [np.array([[(c*f, 0.), (c*f/2., 0.)],
                             [(c*f/2., c), (0., c)]], dtype=float),
                   np.array([[(c*f, 0.), (c*f/2., c)],
                             [(c*f/2, 0.), (0., c)]], dtype=float)]
        
    game = MONFG(payoffs)
    joint_strat = np.array([[1., 0.], [1., 0.]])
    u_tpl = (u_beta([1, 1], beta), u_beta([1, 1], beta))

    is_NE(game, joint_strat, u_tpl)"""
