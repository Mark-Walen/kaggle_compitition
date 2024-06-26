import os.path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

proj_dir = os.path.dirname(os.path.dirname(__file__))


def plots_design(fig, ax):
    fig.patch.set_facecolor('black')
    ax.patch.set_facecolor('black')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.yaxis.set_label_coords(0, 0)
    ax.grid(color='white', linewidth=2)
    # Remove ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    # Remove axes splines
    for i in ['top', 'bottom', 'left', 'right']:
        ax.spines[i].set_visible(False)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')


def onehot_encoding_categories(df: pd.DataFrame):
    categ_cols = df.dtypes[df.dtypes == object]  # filtering by categorical variables
    categ_cols = categ_cols.index.tolist()  # list of categorical fields

    df_enc = pd.get_dummies(df, columns=categ_cols, drop_first=True)  # One hot encoding

    return df_enc


def label_encode(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    return df


def get_top_n_corr(df: pd.DataFrame, y_label: str, n: int = 10):
    """
        Get the top N features with the highest correlation to the target variable.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        y_label (str): The target variable's column name.
        n (int): The number of top features to return. Default is 10.

        Returns:
        pd.Series: A series with the top N features and their correlation values.
        """
    # Ensure the target variable is in the DataFrame
    if y_label not in df.columns:
        raise ValueError(f"{y_label} is not a column in the DataFrame")

    # Calculate the correlation matrix
    numeric_cols = df.select_dtypes(exclude='object').columns
    corr_matrix = df[numeric_cols].corr()

    # Get the correlation values of each feature with the target variable
    target_corr = corr_matrix[y_label]

    # Drop the target variable itself to avoid self-correlation
    target_corr = target_corr.drop(labels=[y_label])

    # Get the top N correlated features
    top_n_corr = target_corr.abs().sort_values(ascending=False).head(n)

    return top_n_corr


def show_pairplot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 9))
    top_corr = get_top_n_corr(df, 'Target', 10)
    top_corr['Target'] = df['Target']
    top_features = top_corr.index.union(['Target'])
    sns.pairplot(df[top_features].sample(frac=0.2), hue="Target")
    print('showing...')
    # plt.show()
    plt.savefig(f'{proj_dir}\\output\\image\\pair plot.2.png')
    

def show_feature_importance(corr: pd.DataFrame, y_label: str):
    """
    Args:
        corr: correlation data
        y_label:
    Returns:

    """
    corr = corr[y_label][:].sort_values(ascending=True).to_frame()
    corr = corr.drop(corr[corr[y_label] > 0.99].index)
    fig, ax = plt.subplots(figsize=(19, 19))

    ax.barh(corr.index, corr[y_label], align='center', color=np.where(corr[y_label] < 0, 'crimson', '#89CFF0'))
    plots_design(fig, ax)
    plt.text(-0.12, 39, "Correlation", size=24, color="grey", fontweight="bold")
    plt.text(0.135, 39, "of", size=24, color="grey")
    plt.text(0.185, 39, y_label, size=24, color="#89CFF0", fontweight="bold")
    plt.text(0.4, 39, "to", size=24, color="grey")
    plt.text(0.452, 39, "Other Features", size=24, color="grey", fontweight="bold")

    # Author
    plt.text(0.9, -7, "@Mark-Walen", fontsize=11, ha="right", color='grey')
    plt.savefig(f'{proj_dir}\\output\\image\\feature_importance.png')


def show_corr_heatmap(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(22, 19))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    sns.heatmap(corr)
    plt.savefig(f'{proj_dir}\\output\\image\\corr.png')


def read_data():
    train = pd.read_csv(f'{proj_dir}\\dataset\\train.csv')
    test = pd.read_csv(f'{proj_dir}\\dataset\\test.csv')

    train_id, test_id = train['id'], test['id']
    del train['id']
    del test['id']

    train = label_encode(train).copy()
    corr = train.corr()
    show_corr_heatmap(corr)
    # show_feature_importance(corr, 'Target')
    # show_pairplot(train)

    train['train'] = 1
    test['train'] = 0
    # df = pd.concat([test, train])


def main():
    read_data()


if __name__ == '__main__':
    main()
