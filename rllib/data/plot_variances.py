import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme()


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

    plt.savefig(f'{save_dir}/beta_by_successful_rescues.png')
    plt.clf()
    pass



if __name__ == "__main__":
    data = pd.read_csv('./data/contribution_dataset.csv')

    line_plot_with_variances(
        data = data,
        save_dir='./data/dominant_contributions',
    )
    