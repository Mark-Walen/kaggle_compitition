import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error


palette = ['#328ca9', '#0e6ea9', '#2c4ea3', '#193882', '#102446']


# Defining plots design
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


def show_corr(df: pd.DataFrame):
    corr = df[df.columns].corr()
    corr = corr['Rings'][:].sort_values(ascending=True).to_frame()
    corr = corr.drop(corr[corr.Rings > 0.99].index)
    fig, ax = plt.subplots(figsize=(9,9))

    ax.barh(corr.index, corr.Rings, align='center', color = np.where(corr['Rings'] < 0, 'crimson', '#89CFF0'))
    plots_design(fig, ax)
    plt.text(-0.12, 39, "Correlation", size=24, color="grey", fontweight="bold")
    plt.text(0.135, 39, "of", size=24, color="grey")
    plt.text(0.185, 39, "Rings", size=24, color="#89CFF0", fontweight="bold")
    plt.text(0.4, 39, "to", size=24, color="grey")
    plt.text(0.452, 39, "Other Features", size=24, color="grey", fontweight="bold")

    # Author
    plt.text(0.9, -7, "@miguelfzzz", fontsize=11, ha="right", color='grey')
    plt.savefig('output/image/corr2.png')


def show_pairplot(df: pd.DataFrame):
    corr = df[df.columns].corr()
    # corr = corr['Rings'][:].sort_values(ascending=True).to_frame()
    top_corr = corr['Rings'].sort_values(ascending=False).head(10).index
    top_corr = top_corr.union(['Rings'])
    
    sns.pairplot(df[top_corr])
    plt.savefig('output/image/pairplot_clean2.png')


def boxplot(df: pd.DataFrame):
    global palette
    NUM_COLS_F = [col for col in df.columns if df[col].dtype == 'float']

    # Define the number of rows and columns for subplots
    num_rows = 4  # 4 rows
    num_cols = 4  # 4 columns

    # Create subplots with appropriate titles
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 17))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through each numerical column and create a box plot
    for i, col in enumerate(NUM_COLS_F[:num_rows * num_cols]):
        sns.boxplot(x=df[col], ax=axes[i], color=palette[i % len(palette)])
        axes[i].set_title(col)

    # Hide empty subplots
    for i in range(len(NUM_COLS_F), num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('output/image/boxplot.png')


def preprocessing(df: pd.DataFrame):
    boxplot(df)
    df1 = df.copy()
    df1 = df1.drop(df1[(df1['Rings'] > 27.5)].index)
    df1 = df1.drop(df1[(df1['Rings'] < 2)].index)
    df1 = df1.drop(df1[(df1['Diameter'] < 0.3) & (df1['Rings'] > 25)].index)
    df1 = df1.drop(df1[(df1['Diameter'] > 0.3) & (df1['Rings'] < 2.5)].index)
    df1 = df1.drop(df1[(df1['Height'] > 0.275)].index)
    df1 = df1.drop(df1[(df1['Length'] < 0.5) & (df1['Rings'] > 25)].index)
    df1 = df1.drop(df1[(df1['Shell weight'] > 1)].index)
    df1 = df1.drop(df1[(df1['Shell weight'] > 0.2) & (df1['Rings'] < 2.5)].index)
    df1 = df1.drop(df1[(df1['Whole weight.1'] > 1.4)].index)
    df1 = df1.drop(df1[(df1['Whole weight.2'] > 0.6)].index)
    print('Outliers removed =' , df.shape[0] - df1.shape[0])

    return df1


def skewed_feature(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(exclude='object').columns

    skew_limit = 0.5
    skew_vals = df[numeric_cols].skew()

    skew_cols = (skew_vals
                .sort_values(ascending=False)
                .to_frame()
                .rename(columns={0:'Skew'})
                .query('abs(Skew) > {0}'.format(skew_limit)))

    for col in skew_cols.index:
        df[col] = boxcox1p(df[col], boxcox_normmax(df[col] + 1))
    
    return df, skew_cols


def skew_visualize(y):
    import matplotlib.ticker as ticker

    # Font
    mpl.rcParams['font.size'] = 10

    # Visualization
    fig, ax = plt.subplots(figsize =(9, 6))
    fig.patch.set_facecolor('black')
    ax.patch.set_facecolor('black')

    sns.histplot(y['Rings'], stat='density', linewidth=0, color = '#ff7f50', kde=True, alpha=0.3)

    # Remove ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Remove axes splines
    for i in ['top', 'bottom', 'left', 'right']:
        ax.spines[i].set_visible(False)

    # Remove grid
    plt.grid(visible=False)

    # Setting thousands with k
    ax.xaxis.set_major_formatter(ticker.EngFormatter())

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    plt.xlabel('Rings', fontsize=11)

    plt.text(7.5, 1.5, "Rings", size=22, color="#ff7f50", fontweight="bold")
    plt.text(11.75, 1.5, "Distribution", size=22, color="grey", fontweight="bold")
    plt.savefig('output/image/skew_y_log.png')


def show_histplot(df: pd.DataFrame):
    figsize = (12, 8)

    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=figsize)

    axs = axs.flatten()

    for i, col in enumerate(df.columns):
        sns.histplot(df, x=col, ax=axs[i], hue='train')
        axs[i].set_title(f'Histogram of {col}')
    
    for i in range(len(df.columns), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


def encoding_categories(df: pd.DataFrame):
    categ_cols = df.dtypes[df.dtypes == object]        # filtering by categorical variables
    categ_cols = categ_cols.index.tolist()                # list of categorical fields

    df_enc = pd.get_dummies(df, columns=categ_cols)   # One hot encoding

    return df_enc


def rmsle_score(y_true, y_pred):
    msle = mean_squared_log_error(abs(y_true), abs(y_pred))
    return np.sqrt(msle)


def Feature_Engineering(df: pd.DataFrame):

    del df['id']
    df["Height"] = df["Height"].clip(upper=0.5,lower=0.01)
    df["Volume"] = df["Length"]*df["Diameter"]*df["Height"]
    df["Density"] = df["Whole weight"]/df["Volume"]
    # df['Diameter_Length_ratio'] = df['Diameter'] / df['Length']
    df['Height_Length_ratio'] = df['Height'] / df['Length']
    df['Shell_Whole_weight_ratio'] = df['Shell weight'] / df['Whole weight']
    df['Mean_weights'] = (df['Whole weight']+df['Whole weight.1']+df['Whole weight.2'])/3 
    df = df.reset_index(drop=True)
    
    # Return Data
    return df


def lasso_select_feature_importance(lasso_tuned, test):
    # Selecting features importance
    coefs = pd.Series(lasso_tuned.coef_, index = test.columns)

    lasso_coefs = pd.concat([coefs.sort_values().head(10),
                            coefs.sort_values().tail(10)])

    lasso_coefs = pd.DataFrame(lasso_coefs, columns=['importance'])

    # Visualization
    fig, ax = plt.subplots(figsize =(11, 9))

    ax.barh(lasso_coefs.index, lasso_coefs.importance, align='center', 
            color = np.where(lasso_coefs['importance'] < 0, 'crimson', '#89CFF0'))

    plots_design(fig, ax)

    plt.text(-0.22, 20.5, "Feature Importance", size=24, color="grey", fontweight="bold")
    plt.text(-0.063, 20.5, "using", size=24, color="grey")
    plt.text(-0.0182, 20.5, "Lasso Model", size=24, color="#89CFF0", fontweight="bold")

    # Author
    plt.text(6, -2, "@Mark-Walen", fontsize=12, ha="right", color='grey')
    plt.show()


def lasso(X, y):
    scaler = Normalizer(norm='l2')
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    alpha = np.geomspace(1e-8, 1e-5, num=1000)
    lasso_cv_model = LassoCV(alphas = alpha, cv = 10, max_iter = 100000).fit(X_train, y_train)
    lasso_tuned = Lasso(max_iter = 100000).set_params(alpha=lasso_cv_model.alpha_).fit(X_train, y_train)
    print('The Lasso II:')
    print("Alpha =", lasso_cv_model.alpha_)
    print("RMSLE =", rmsle_score(y_test, lasso_tuned.predict(X_test)))
    print("score: ", lasso_tuned.score(X_test, y_test))

    return lasso_tuned


def model_train():
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')

    train['train'] = 1
    test['train'] = 0
    
    train_id, test_id = train['id'], test['id']
    df = pd.concat([test, train])
    df = Feature_Engineering(df).copy()
    df = preprocessing(df[df['train'] == 1])

    y = df['Rings'].to_frame().dropna(axis=0)
    df =  df.drop('Rings', axis=1)
    
    # df_enc = encoding_categories(df)
    # show_histplot(df_enc)
    # train_id, X, y = preprocessing(train)
    
    # Combining train and test for data cleaning
    # df, _ = skewed_feature(df)
    # df = encoding_categories(df)
    # _X = df[df['train'] == 1]
    # _test = df[df['train'] == 0]

    # X = _X.copy()
    # test = _test.copy()
    # X.drop(['train'], axis=1, inplace=True)
    # test.drop(['train'], axis=1, inplace=True)


    # y['Rings'] = np.log1p(y['Rings'])
    # y = y['Rings']
    
    # model = lasso(X, y)
    # lasso_select_feature_importance(model, test)
    # y_hat = model.predict(test)
    # res = pd.DataFrame(
    #     {'id': test_id, 'Rings': y_hat}
    # )
    # res.to_csv('dataset/submit.csv', index=None)
    # X['Rings'] = y
    # show_corr(X)


def test(theta):
    data_test = pd.read_csv('dataset/test.csv')


def main():
    model_train()


if __name__ == '__main__':
    main()

