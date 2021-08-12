import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
sns.set_theme()


def plot_confusion_matrix(
    allocations, 
    rescue_amounts, 
    save_dir,
    title,
    n_rows = 8,
    n_cols = 8
    ):

    confusion_matrix = np.zeros((n_rows, n_cols))
    for actual,optimal in zip(allocations, rescue_amounts):
            if actual < n_rows:
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

    plt.savefig(
        f'{save_dir}/{title}_with_labels.png',
    )

    ax1.set(xlabel='', ylabel='')
    ax1.set_title('')
    plt.savefig(
        f'{save_dir}/{title}_without_labels.png',
        bbox_inches = 'tight',
        pad_inches = 0.2
    )
    plt.clf()
    plt.close(fig)


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
    plt.close(fig)


def line_plot_with_variances(
    data,
    save_dir,
    ):

    df = pd.DataFrame(data)
    sns.lineplot(data=df, x='Beta', y='Dominant Contributions', hue='Scenario')
    plt.title("Beta x Dominant Contributions")
    plt.savefig(
        f'{save_dir}/beta_by_dominant_contributions.png', 
        bbox_inches = 'tight',
        pad_inches = 0
    )
    plt.clf()

    sns.lineplot(data=df, x='Beta', y='Percentage of Rescue Amount', hue='Scenario')
    plt.title("Beta x Percentage of Rescue Amount")
    plt.savefig(
        f'{save_dir}/beta_by_percentage_of_rescue.png',
        bbox_inches = 'tight',
        pad_inches = 0
    )
    plt.clf()


    sns.lineplot(data=df, x='Beta', y='Successful Rescues', hue='Scenario')
    plt.title("Beta x Successful Rescues")
    plt.savefig(f'{save_dir}/beta_by_successful_rescues.png',
        bbox_inches = 'tight',
        pad_inches = 0
    )
    plt.clf()

    sns.lineplot(data=df, x='Beta', y='Total Contribution', hue='Scenario')
    plt.title("Beta x Total Contributions as percentage of rescue amount")
    plt.savefig(
        f'{save_dir}/beta_by_total_contributions.png',
        bbox_inches = 'tight',
        pad_inches = 0
    )
    plt.clf()


    plt.close()

    pass
