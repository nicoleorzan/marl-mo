import argparse
import numpy as np
from src.experiments.cartpole.cartpole_train_ppo_single_obj import train_ppo_single_obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int)
        
    parser.add_argument('--num_epochs', type=int, default=50000) # sarebbe total_steps
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--wandb_mode', type=str, choices = ["online", "offline"], default="offline")
    
    args = parser.parse_args()
    
    print("\n\n===============>CARTPOLE PPO!!\n\n\n")
    train_ppo_single_obj(args)
