import os.path

import pandas as pd

from sklearn.preprocessing import LabelEncoder

proj_path = os.path.dirname(os.path.dirname(__file__))


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


def read_data():
    train = pd.read_csv(f'{proj_path}\\dataset\\train.csv')
    test = pd.read_csv(f'{proj_path}\\dataset\\test.csv')

    train = label_encode(train).copy()
    train['train'] = 1
    test['train'] = 0

    train_id, test_id = train['id'], test['id']
    del train['id']
    del test['id']

    df = pd.concat([test, train])


def main():
    read_data()


if __name__ == '__main__':
    main()
