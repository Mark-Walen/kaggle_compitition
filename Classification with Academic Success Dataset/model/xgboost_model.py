import gc
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

proj_dir = os.path.dirname(os.path.dirname(__file__))

palette = ['#328ca9', '#0e6ea9', '#2c4ea3', '#193882', '#102446']


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
    fig, ax = plt.subplots(figsize=(25, 21))
    top_corr = get_top_n_corr(df, 'Target', 15)
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


def show_boxplot(df: pd.DataFrame, exclude_cols=None, show: bool = ...):
    global palette

    cols = df.select_dtypes(include='number').columns
    if exclude_cols:
        cols = [col for col in cols if col not in exclude_cols]

    # Define the number of rows and columns for subplots
    num_rows = 5  # 4 rows
    num_cols = 4  # 4 columns

    # Create subplots with appropriate titles
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 17))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Customize flier properties (outliers)
    flier_props = dict(marker='o', markerfacecolor='red', markersize=5, linestyle='none')

    # Loop through each numerical column and create a box plot
    for i, col in enumerate(cols[:num_rows * num_cols]):
        sns.boxplot(x=df[col], ax=axes[i], color=palette[i % len(palette)], flierprops=flier_props)
        axes[i].set_title(col)

    # Hide empty subplots
    for i in range(len(cols), num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(f'{proj_dir}\\output\\image\\boxplot_2.png')


def feature_engineering(df: pd.DataFrame):
    pass


def drop_outlier(df: pd.DataFrame):
    # print(df.info())
    df1 = df.copy()
    prev_count = df1.shape[0]
    # df1 = df1.drop(df1[df1['Application order'] > 3].index)
    df1 = df1.drop(df1[df1['Age at enrollment'] > 60].index)
    # df1 = df1.drop(df1[df1['Age at enrollment'] > 30].index)
    df1 = df1.drop(df1[(df1['Curricular units 1st sem (enrolled)'] > 15) | (df1['Curricular units 1st sem (enrolled)'] < 4)].index)
    df1 = df1.drop(df1[(df1['Curricular units 1st sem (evaluations)'] > 30) | (df1['Curricular units 1st sem (evaluations)'] < 2)].index)
    df1 = df1.drop(df1[df1['Curricular units 1st sem (approved)'] > 20].index)
    # df1 = df1.drop(df1[(df1['Curricular units 1st sem (grade)'] < 7.5) | (df1['Curricular units 1st sem (grade)'] > 17.5)].index)
    df1 = df1.drop(df1[(df1['Curricular units 1st sem (grade)'] < 7.5)].index)
    df1 = df1.drop(df1[df1['Curricular units 2nd sem (approved)'] > 12.5].index)
    df1 = df1.drop(df1[(df1['Curricular units 2nd sem (enrolled)'] > 15) | (df1['Curricular units 2nd sem (enrolled)'] < 4)].index)
    df1 = df1.drop(df1[(df1['Curricular units 2nd sem (evaluations)'] > 28) |
                       (df1['Curricular units 2nd sem (evaluations)'] < 2)].index)
    df1 = df1.drop(df1[df1['Curricular units 2nd sem (grade)'] < 7.5].index)
    # print('Total removed: {}'.format(prev_count - df1.shape[0]))
    return df1


def skewed_feature(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(exclude='object').columns

    skew_limit = 0.5
    skew_vals = df[numeric_cols].skew()

    skew_cols = (skew_vals
                 .sort_values(ascending=False)
                 .to_frame()
                 .rename(columns={0: 'Skew'})
                 .query('abs(Skew) > {0}'.format(skew_limit)))

    for col in skew_cols.index:
        if col == 'Height_log':
            continue
        df[col] = boxcox1p(df[col], boxcox_normmax(df[col] + 1))

    return df, skew_cols


def target_predictor(X, y, test, iterations, model, model_name):
    df_preds = pd.DataFrame()
    df_preds_x = pd.DataFrame()
    k = 1
    splits = iterations
    avg_score = 0

    # CREATING STRATIFIED FOLDS
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=200)
    print('\nStarting KFold iterations...')
    for train_index, test_index in skf.split(X, y):

        df_X = X.iloc[train_index, :]
        df_y = y[train_index]
        val_X = X.iloc[test_index, :]
        val_y = y[test_index]

        # FITTING MODEL
        model.fit(df_X, df_y)

        # PREDICTING ON VALIDATION DATA
        col_name = model_name + 'xpreds_' + str(k)
        preds_x = pd.Series(model.predict(val_X))
        df_preds_x[col_name] = pd.Series(model.predict(X))

        # CALCULATING ACCURACY
        acc = accuracy_score(val_y, preds_x)
        print('Iteration:', k, '  accuracy_score:', acc)
        if k == 1:
            score = acc
            best_model = model
            preds = pd.Series(model.predict(test))
            col_name = model_name + 'preds_' + str(k)
            df_preds[col_name] = preds
        else:
            preds1 = pd.Series(model.predict(test))
            preds = preds + preds1
            col_name = model_name + 'preds_' + str(k)
            df_preds[col_name] = preds1
            if score < acc:
                score = acc
                best_model = model
        avg_score = avg_score + acc
        k = k + 1
    print('\n Best score:', score, ' Avg Score:', avg_score / splits)
    # TAKING AVERAGE OF PREDICTIONS
    preds = preds / splits

    # print('Saving test and train predictions per iteration...')
    # df_preds.to_csv(model_name + '.csv', index=False)
    # df_preds_x.to_csv(model_name + '_.csv', index=False)
    x_preds = df_preds_x.mean(axis=1)
    del df_preds, df_preds_x
    gc.collect()
    return preds, best_model, x_preds


def read_data():
    train = pd.read_csv(f'{proj_dir}\\dataset\\train.csv')
    test = pd.read_csv(f'{proj_dir}\\dataset\\test.csv')

    # train = label_encode(train).copy()
    # corr = train.corr()
    # show_corr_heatmap(corr)
    # show_feature_importance(corr, 'Target')
    # show_boxplot(train,
    #              exclude_cols=["Marital status", "Application mode", "Daytime/evening attendance",
    #                            "Previous qualification", "Mather's qualification", "Father's qualification",
    #                            "Mother's occupation", "Father's occupation", "Displaced", "Debtor",
    #                            "Tuition fees up to date", "Scholarship holder", "International",
    #                            "Curricular units 1st sem (without evaluations)", "Curricular units 1st sem (credited)",
    #                            "Curricular units 2nd sem (without evaluations)", "Curricular units 2nd sem (credited)",
    #                            "Unemployment rate", "Inflation rate", "GDP", "Target", "Education special need"],
    #              show=False)
    # train = drop_outlier(train).copy()
    train_id, test_id = train['id'], test['id']
    del train['id']
    del test['id']
    # show_pairplot(train)
    print(train.shape)

    df_dtype = pd.DataFrame()
    df_dtype['cols'] = train.columns[:-1]
    df_dtype['data_type'] = list([train[col].dtype for col in train.columns[:-1]])
    df_dtype['unique_val'] = list([len(train[col].unique()) for col in train.columns[:-1]])
    df_dtype['val_type'] = list(['conti' if x == 'float32' else 'categ' for x in df_dtype['data_type']])

    # Identifying binary features
    categ_cols_bin = df_dtype['cols'][(df_dtype.val_type == 'categ') & (df_dtype.unique_val <= 2)]

    # Identifying categorical features with categories greater than 10
    categ_cols_nonbin = df_dtype['cols'][(df_dtype.val_type == 'categ') & (df_dtype.unique_val > 2)]

    # Identifying continuous features
    conti_cols = df_dtype['cols'][df_dtype.val_type == 'conti']

    # train['train'] = 1
    # test['train'] = 0
    # df = pd.concat([test, train])

    # y = train['Target'].to_frame().dropna(axis=0)
    # df = train.drop('Target', axis=1)

    # Combining train and test for data cleaning
    # df, _ = skewed_feature(df)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(train[list(categ_cols_nonbin) + list(conti_cols)]),
                     columns=list(categ_cols_nonbin) + list(conti_cols))
    X = pd.concat([X, train[categ_cols_bin]], axis=1)
    # for col in categ_cols_bin:
    #     X[col] = train[col]
    test_non_bin = pd.DataFrame(scaler.transform(test[list(categ_cols_nonbin) + list(conti_cols)]),
                        columns=list(categ_cols_nonbin) + list(conti_cols))
    test = pd.concat([test_non_bin, test[categ_cols_bin]], axis=1)
    y = pd.DataFrame([0 if x == 'Graduate' else 1 if x == 'Dropout' else 2 for x in train['Target']]).values
    print(X.shape, y.shape, test.shape)
    xgb_params = {'objective': 'multiclass',
                  'metric': 'accuracy',
                  'learning_rate': 0.5,
                  'max_depth': 3,
                  'min_child_weight': 5,
                  'n_estimators': 100,
                  'n_jobs': 1,
                  'subsample': 0.6500000000000001
                  }
    model = xgb.XGBClassifier(**xgb_params)
    xgb_predictions, best_xgb_model, XGBpreds = target_predictor(X, y, test, 10, model, 'XGB')
    predict = model.predict(test)
    res = pd.DataFrame(
        {'id': test_id, 'Target': predict}
    )
    res['Target'] = res['Target'].map({0: 'Graduate', 1: 'Dropout', 2: 'Enrolled'})
    res.to_csv(f'{proj_dir}\\dataset\\submit.csv', index=None)


def main():
    read_data()


if __name__ == '__main__':
    main()
