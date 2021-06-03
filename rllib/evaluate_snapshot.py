import ray
import json
from utils import get_args
from rllib_train import setup
from ray.rllib.agents.dqn import DQNTrainer
from env import Volunteers_Dilemma

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.text import Text

import os

def plot(
    actual_allocations, 
    optimal_allocations, 
    save_dir,
    title="Single Agent allocation; 1e3 eval, 1e6 training episodes",
    maximum_allocation = 7,):

    n_rows = maximum_allocation + 2
    n_cols = maximum_allocation + 2
    n_cells = n_rows * n_cols
    confusion_matrix = np.zeros((n_rows, n_cols))
    for actual,optimal in zip(actual_allocations, optimal_allocations):
        if actual >=0 and actual <=7:
            confusion_matrix[int(actual),int(optimal)] += 1

    for i in range(confusion_matrix.shape[0]):
        confusion_matrix[i,-1] = confusion_matrix[i,:-1].sum()
        confusion_matrix[-1,i] = confusion_matrix[:-1,i].sum()

    print(confusion_matrix)


    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in range(n_rows-1)] + ['Total'],
                      columns = [i for i in range(n_cols-1)] + ['Total'])

    plt.figure(figsize = (10,7))
    plt.title(title)
    plt.ylabel("Actual Allocation")
    plt.xlabel("Optimal Allocation")
    ax = sn.heatmap(df_cm, annot=True, fmt='g', cmap="Blues", annot_kws={"fontsize":8}, cbar=False)
    ax.set(xlabel='Optimal Allocation', ylabel='Actual Allocation')

    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # make colors of the last column white
    facecolors[np.arange(n_rows-1,n_cells,n_cols)] = np.array([1,1,1,1])
    facecolors[np.arange(n_cells-n_cols,n_cells)] = np.array([1,1,1,1])

    quadmesh.set_facecolors = facecolors

    # set color of all text to black
    for i in ax.findobj(Text):
        i.set_color('black')

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 

    plt.savefig(f'{save_dir}/{title}.png')
    plt.clf()


def plot_hist(data, title, save_dir):
    sn.set_theme()
    sn.histplot(data=data, shrink =0.8, bins=30)
    plt.xlabel("Allocations")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(f'{save_dir}/{title}_hist.png')
    plt.clf()


def jointplot(x, y, title, save_dir):

    x_lim_max = np.max([np.max(y),10])
    y_lim_max = np.max([np.max(y),10])

    sn.set_theme()
    g = sn.jointplot(
        x = x.flatten(), 
        y = y.flatten(),
        xlim = [0, x_lim_max],
        ylim = [0, y_lim_max],
    )
    g.set_axis_labels(xlabel="agent 0", ylabel="agent 1")
    plt.savefig(f'{save_dir}/{title}_jointplot.png')
    plt.clf()


    

if __name__ == "__main__":
    args = get_args()
    ray.init(local_mode = args.local_mode)
    config, env_config, stop = setup(args)

    config['explore'] = False

    checkpoints = [50,100,150,200]
    n_rounds = 100
    

    with open('results_dictionary.json') as f:
        data = f.read()
    dictionary = json.loads(data)
    
    path = f"/itet-stor/bryayu/net_scratch/results/{dictionary[str(args.experiment_number)]}"

    if not os.path.exists(f'./data/checkpoints/{args.experiment_number}'):
        os.makedirs(f'./data/checkpoints/{args.experiment_number}')
    

    for checkpoint in checkpoints:

        save_dir = f'./data/checkpoints/{args.experiment_number}/{checkpoint}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        agent = DQNTrainer(config=config, env=Volunteers_Dilemma)
        agent.restore(f"{path}/checkpoint_{checkpoint}/checkpoint-{checkpoint}")

        # instantiate env class
        env = Volunteers_Dilemma(env_config)

        agent_0_actions = []
        agent_1_actions = []
        optimal_allocation = []

        for i in range(n_rounds):
            a_0 = []
            a_1 = []
            obs = env.reset()
            for round in range(env_config.get('number_of_negotiation_rounds')):
                actions = {}
                
                action_0 = agent.compute_action(obs[0], policy_id='policy_0')
                actions[0] = action_0
                a_0.append(action_0)
                
                if env_config.get('n_agents') == 2:
                    action_1 = agent.compute_action(obs[1], policy_id='policy_1')
                    actions[1] = action_1
                    a_1.append(action_1)


                obs, _, _, _ = env.step(actions)

            agent_0_actions.append(a_0)
            agent_1_actions.append(a_1)
            optimal_allocation.append(-obs[0]['real_obs'][2])
        
        pass

        agent_0_actions = np.array(agent_0_actions)
        agent_1_actions = np.array(agent_1_actions)
        optimal_allocations = np.array(optimal_allocation)

        for round in range(env_config.get('number_of_negotiation_rounds')):
            a0_actions = agent_0_actions[:,round]
            plot(a0_actions, optimal_allocation, save_dir=save_dir, title=f"Checkpoint {checkpoint}, Agent 0, Round {round}")
            plot_hist(a0_actions, save_dir=save_dir, title=f"Checkpoint {checkpoint}, Agent 0, Round {round}")

            if env_config.get('n_agents') == 2 :
                a1_actions = agent_1_actions[:,round]
                plot(a1_actions, optimal_allocation, save_dir= save_dir, title=f"Checkpoint {checkpoint}, Agent 1, Round {round}")
                plot_hist(a1_actions,save_dir=save_dir, title=f"Checkpoint {checkpoint}, Agent 1, Round {round}")

                jointplot(a0_actions, a1_actions, save_dir=save_dir, title=f"Checkpoint {checkpoint} Round {round}")


    ray.shutdown()
