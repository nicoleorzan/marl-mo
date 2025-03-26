import torch
import numpy as np

COINS = 4

def GGF(x, w):
    #print("w=", w)
    #print("x=", x)
    x_up = x.sort(dim=0)[0]
    #print("x_up=", x_up)
    #ggf = torch.inner(w, x_up)
    #print("act0=", x_up[0][0]*w[0] + x_up[1][0]*w[1] + x_up[2][0]*w[2])
    #print("act1=", x_up[0][1]*w[0] + x_up[1][1]*w[1] + x_up[2][1]*w[2])
    ggf = torch.matmul(w, x_up)
    #print("ggf=", ggf)
    return ggf

def reward_function(f, a, c):
    common_pot = torch.sum(torch.Tensor([c*a[i] for i in range(2)]))
    # compute reward only for agent 0
    group_reward = common_pot/2*f 
    individual_reward = c-c*a[0]

    reward = torch.Tensor([ group_reward, individual_reward[0] ])
    return reward

def study_of_ggf(f, a, wi, c):
    rew = reward_function(f, a, c)

    #for wi in w:
    weight = torch.Tensor([1.-wi,wi])
    #print("rew=", rew)
    #print("weight=", weight)
    ggf = GGF(rew, weight)
    print("r=", rew,"r ord=",rew.sort()[0],"act=", a, "ggf=", ggf)
    return ggf


actions = [[torch.tensor([0]), torch.Tensor([0])], \
           [torch.tensor([0]), torch.Tensor([1])], \
            [torch.tensor([1]), torch.Tensor([0])], \
            [torch.tensor([1]), torch.Tensor([1])]]

actions_single = [torch.tensor([0]), torch.Tensor([1])]

F = [100.5]#, 1.0, 1.5, 2.5, 3.5]
weights = torch.linspace(0.0, 1.0, 10)
for f in F:
    print("\nFor a game with f=", f)
    for wi in weights:
        if (1.-wi > wi):
            print("weight=", torch.Tensor([1.-wi,wi]))
            list_vals = []
            for ai in actions_single:
                vals = []
                for aj in actions_single:
                    a = [ai, aj]
                    ggf = study_of_ggf(f, a, wi, COINS)
                    vals.append(ggf)
                list_vals.append(np.mean(vals))
            if ( list_vals[1] - list_vals[0] > 0.):
                print("average ggf D, C=", list_vals,"val[C] - val[D]=", list_vals[1] - list_vals[0], "better COOPERATE")
            else:
                print("average ggf D, C=", list_vals,"val[C] - val[D]=", list_vals[1] - list_vals[0], "better DEFECT")

                    
