import pandas as pd
import numpy as np
import joblib

from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score

model = tree.DecisionTreeClassifier()


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


def train():
    df = pd.read_csv('dataset/train.csv')
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Fare']
    x_cols = ['Sex', 'Pclass', 'Fare']
    y_cols = ['Survived']
    
    df = prepare_data(df)
    _df = df[cols]

    _df_norm = _df.apply(lambda x: (x - np.mean(x)) / (np.std(x)), axis=0)
    X = np.array(_df_norm[x_cols], dtype=np.float64)
    y = np.array(df[y_cols], dtype=np.int32)

    kf = KFold(n_splits=10, random_state=None)
    _score = 0
    _model = None
    for train_idx, test_idx in kf.split(X):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        _m = model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        print('RMSE: ', mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
        print('score: ', score)

        if _score < score:
            _score = score
            _model = _m

    joblib.dump(_model, 'output/decision_tree.pkl')


def predict(X, y):
    _model = joblib.load('output/decision_tree.pkl')
    y_pred = _model.predict(X)

    print('score: ', accuracy_score(y, y_pred))
    return y_pred


def test():
    data_test = pd.read_csv('dataset/test.csv')
    y_test = pd.read_csv('dataset/gender_submission.csv')['Survived']
    data_test = prepare_data(data_test)
    x_cols = ['Sex', 'Pclass', 'Fare']
    _df = data_test[x_cols]

    # _df_norm = (_df - _df.mean()) / (_df.std())
    _df_norm = _df.apply(lambda x: (x - np.mean(x)) / (np.std(x)), axis=0)
    x_train = np.array(_df_norm[x_cols], dtype=np.float64)
    y_hat = predict(x_train, y_test)

    y_hat = y_hat.ravel()
    res = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'], 'Survived': y_hat.astype(np.int32)})
    res.to_csv('dataset/submit.csv', index=None)


def main():
    test()


if __name__ == '__main__':
    main()
