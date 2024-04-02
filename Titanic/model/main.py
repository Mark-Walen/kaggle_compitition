import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics


def mini_batch_gradient_descent(X, y, theta, learning_rate=0.01, batch_size=32, epochs=100):
    m = len(y)
    n_batches = m // batch_size

    for epoch in range(epochs):
        # Shuffle the dataset
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, n_batches):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Compute gradient
            error = X_batch.dot(theta) - y_batch
            gradient = X_batch.T.dot(error) / batch_size

            # Update parameters
            theta -= learning_rate * gradient
            # print(gradient.shape[0], gradient.shape[1])

    return theta


def prepare_data(df: pd.DataFrame):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    ans = df['Embarked'].value_counts()
    fill_str = ans.idxmax()
    df['Embarked'] = df['Embarked'].fillna(fill_str)

    df.loc[df["Sex"] == "male", "Sex"] = 0
    df.loc[df["Sex"] == "female", "Sex"] = 1

    df.loc[df['Embarked'] == 'C', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 2

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    return df


def predict(X, theta):
    y_hat = X.dot(theta)
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0

    return y_hat


def train():
    df = pd.read_csv('dataset/train.csv')
    cols = ['Pclass', 'Sex', 'Age', 'SibSp',
            'Parch', 'Embarked', 'Fare', 'Survived']
    x_cols = ['Pclass', 'Sex', 'Embarked', 'Fare']
    y_cols = ['Survived']

    df = prepare_data(df)
    _df = df[cols]

    # _df_norm = (_df - _df.mean()) / (_df.std())
    _df_norm = _df.apply(lambda x: (x - np.mean(x)) / (np.std(x)), axis=0)
    X = np.array(_df_norm[x_cols], dtype=np.float64)
    y = np.array(_df_norm[y_cols], dtype=np.float64)

    kf = KFold(n_splits=10, random_state=None)
    _score = 0
    _theta = None
    for train_idx, test_idx in kf.split(X):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        theta = np.zeros(x_train.shape[1]).reshape(-1, 1)
        theta = mini_batch_gradient_descent(x_train, y_train, theta)
        y_hat = predict(x_test, theta)

        y_test = np.array(df[y_cols].loc[test_idx])
        score = metrics.accuracy_score(y_test, y_hat)
        if _score < score:
            _score = score
            _theta = theta
        print('accuracy_score', score)

    return _theta


def test(theta):
    data_test = pd.read_csv('dataset/test.csv')
    y_test = pd.read_csv('dataset/gender_submission.csv')['Survived']
    data_test = prepare_data(data_test)
    x_cols = ['Pclass', 'Sex', 'Embarked', 'Fare']
    _df = data_test[x_cols]

    # _df_norm = (_df - _df.mean()) / (_df.std())
    _df_norm = _df.apply(lambda x: (x - np.mean(x)) / (np.std(x)), axis=0)
    x_train = np.array(_df_norm[x_cols], dtype=np.float64)
    y_hat = predict(x_train, theta)

    y_hat = y_hat.ravel()
    res = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'], 'Survived': y_hat.astype(np.int32)})
    res.to_csv('dataset/submit.csv', index=None)
    score = metrics.accuracy_score(y_test, y_hat)
    print('accuracy_score', score)


def main():
    # print(train())
    # theta = np.array([[-0.26441397],
    #                   [0.49143954],
    #                   [-0.08471867],
    #                   [-0.00686108]])
    test(train())


if __name__ == '__main__':
    main()

"""
            Pclass       Sex       Age     SibSp     Parch  Embarked      Fare  Survived
Pclass    1.000000 -0.131900 -0.339693  0.083081  0.018405  0.162098 -0.549677 -0.338481
Sex      -0.131900  1.000000 -0.081240  0.114631  0.245521 -0.108262  0.182308  0.543351
Age      -0.339693 -0.081240  1.000000 -0.233261 -0.172638 -0.018920  0.096657 -0.065098
SibSp     0.083081  0.114631 -0.233261  1.000000  0.414788  0.068230  0.159851 -0.035322
Parch     0.018405  0.245521 -0.172638  0.414788  1.000000  0.039741  0.216346  0.081704
Embarked  0.162098 -0.108262 -0.018920  0.068230  0.039741  1.000000 -0.224867 -0.167675
Fare     -0.549677  0.182308  0.096657  0.159851  0.216346 -0.224867  1.000000  0.257490
Survived -0.338481  0.543351 -0.065098 -0.035322  0.081704 -0.167675  0.257490  1.000000
"""
