import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.text import Text

def plot(
    actual_allocations, 
    optimal_allocations, 
    title="Single Agent allocation; 1e3 eval, 1e6 training episodes",
    maximum_allocation=7, ):

    n_rows = maximum_allocation + 2
    n_cols = maximum_allocation + 2
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
    facecolors[np.arange(7,64,8)] = np.array([1,1,1,1])
    facecolors[np.arange(64-8,64)] = np.array([1,1,1,1])

    quadmesh.set_facecolors = facecolors

    # set color of all text to black
    for i in ax.findobj(Text):
        i.set_color('black')

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 

    plt.show()
    plt.savefig(f'{title}.png')


if __name__ == "__main__":
    actual_allocations = pd.read_csv('./data/experiments/3/actual_allocation.csv')['Value'].tolist()[:500]
    optimal_allocations = pd.read_csv('./data/experiments/3/optimal_allocation.csv')['Value'].tolist()[:500]

    plot(actual_allocations, optimal_allocations, title="Checking if 102 at position 7,1 is a bug; fixed seed 2")    

    pass
