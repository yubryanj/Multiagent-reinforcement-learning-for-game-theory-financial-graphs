import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
sns.set_theme()


# def plot_uniformly_mixed(
#     path,
#     n_rows = 7,
#     n_cols = 4
#     ):
#     """
#     TODO: Write me
#     """

#     # Read in the dataset
#     df = pd.read_csv(f'{path}/experimental_data.csv')

#     # Extract one dataset per sub scenario
#     sub_scenarios = df.sub_scenarios.unique()
#     datasets = []
#     for sub_scenario in sub_scenarios:
#         datasets.append(df.loc[df['sub_scenarios'] == sub_scenario])

#     if not os.path.exists(f'{path}/scenarios'):
#         os.makedirs(f'{path}/scenarios')

    
#     #Generate one confusion matrix per scenario
#     for dataset, subscenario in zip(datasets, sub_scenarios):
        
#         plot_confusion_matrix(
#             actual_allocations = dataset['agent 0 actions'],
#             optimal_allocations= dataset['rescue_amount'],
#             save_dir = f'{path}/scenarios/',
#             title=f'Uniformly mixed - agent 0 - {subscenario}',
#         )

#         plot_confusion_matrix(
#             actual_allocations = dataset['agent 1 actions'],
#             optimal_allocations= dataset['rescue_amount'],
#             save_dir = f'{path}/scenarios/',
#             title=f'Uniformly mixed - agent 1 - {subscenario}',
#         )



def plot_confusion_matrix(
    allocations, 
    rescue_amounts, 
    save_dir,
    title,
    n_rows = 7,
    n_cols = 7
    ):

    confusion_matrix = np.zeros((n_rows, n_cols))
    for actual,optimal in zip(allocations, rescue_amounts):
            confusion_matrix[int(actual),int(optimal)] += 1

    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in range(n_rows)],
                      columns = [i for i in range(n_cols)])

    sns.set(font_scale=1.25)

    fig = plt.figure(figsize = (20,15))
    ax1 = plt.subplot2grid((20,20), (0,0), colspan=17, rowspan=17)
    ax2 = plt.subplot2grid((20,20), (17,0), colspan=17, rowspan=2)
    ax3 = plt.subplot2grid((20,20), (0,17), colspan=2, rowspan=17)

    sns.heatmap(df_cm, ax=ax1, annot=True, fmt='g', cmap="Blues", cbar=False)
    
    ax1.xaxis.tick_top()
    ax1.set(xlabel='Rescue Amount', ylabel='Actual Allocation')
    ax1.set_title(title)
    ax1.xaxis.set_label_position('top')

    sns.heatmap((pd.DataFrame(df_cm.sum(axis=0))).transpose(), ax=ax2,  annot=True, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
    sns.heatmap(pd.DataFrame(df_cm.sum(axis=1)), ax=ax3,  annot=True, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)

    plt.savefig(f'{save_dir}/{title}.png')
    plt.clf()


def plot_table(
        data,
        save_dir,
        title, 
    ):
    fig, ax =plt.subplots(1,1)    
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data,loc="center")
    ax.set_title(title)

    plt.savefig(f'{save_dir}/{title}.png')
    plt.clf()


def line_plot_with_variances(
    data,
    save_dir,
    ):

    df = pd.DataFrame(data)
    sns.lineplot(data=df, x='beta', y='dominant_contributions')
    plt.title("Beta x Dominant Contributions")

    plt.savefig(f'{save_dir}/Beta_by_Dominant_Contributions.png')
    plt.clf()


    sns.lineplot(data=df, x='beta', y='percentage_of_rescue_amount')
    plt.title("Beta x Percentage of rescue amount")

    plt.savefig(f'{save_dir}/Beta_by_percentage_of_rescue.png')
    plt.clf()


    sns.lineplot(data=df, x='beta', y='successful_rescues')
    plt.title("Beta x rescues")

    plt.savefig(f'{save_dir}/Beta_by_successful_rescues.png')
    plt.clf()
    pass
