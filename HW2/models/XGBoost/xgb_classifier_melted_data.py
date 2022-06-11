import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from HW2.models.print_results import print_results


def get_x_y(df, x_filter, y_filter):
    x = df[x_filter]  # Features
    y = df[y_filter]  # Target variables
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    y = y.astype('float')
    x = x.astype('float')
    return x, y


def convert_category_variables(df):
    season_variables_lst = ['born_in_season', 'fill_in_season']
    for category_variable in season_variables_lst:
        mapping = {"winter": 0.0, "fall": 1.0, "summer": 2.0, "spring": 3.0}
        df[category_variable] = df[category_variable].map(mapping)
    profil_mapping = {21: 0.0, 24: 1.0, 25: 2.0, 45: 3.0, 64: 4.0, 72: 5.0, 82: 6.0, 97: 7.0}
    df['profil'] = df['profil'].map(profil_mapping)
    choice_mapping = {1.0: 1.0, 2.0: 1.0, 3.0: 2.0, 4.0: 3.0, 5.0: 3.0, 100.0: 100.0}
    if 'choice_value' in df:
        df['choice_value'] = df['choice_value'].map(choice_mapping)
    return df


def manual_category_convert(df):
    df = convert_category_variables(df)
    mapping = {"general": 0.0, "air": 1.0, "navy": 2.0, "ground": 3.0}
    df['force'] = df['force'].map(mapping)
    text_cols = ['is_technological', 'is_physical', 'is_leadership']
    txt_mapping = {"no": 0.0, "yes": 1.0}
    for col in text_cols:
        df[col] = df[col].map(txt_mapping)
    return df


def split_df_to_validation_and_training(df, y_filter, x_filter):
    print("split df to validation and training sets")
    x, y = get_x_y(df, x_filter, y_filter)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_test = np.nan_to_num(x_test)
    x_train = np.nan_to_num(x_train)
    y_train = np.nan_to_num(y_train)
    y_test = np.nan_to_num(y_test)
    return x_train, x_test, y_train, y_test


def label_encoding(df):
    le = preprocessing.LabelEncoder()
    cols_for_transform = ['role', 'cluster', 'role_name']
    for col in cols_for_transform:
        df[col] = le.fit_transform(df[col])


def prepare_df():
    df = pd.read_csv("../../csv_files/melt_final_manila_data.csv", encoding="UTF-8")
    print("finished to read data")
    df = manual_category_convert(df)
    print("finished to convert category variables")
    df = clean_dataset(df)
    print("finished to clean data")
    label_encoding(df)
    print("df is ready!")
    return df


def clean_dataset(df):
    df = df.replace((np.inf, -np.inf, np.nan, "-", "..", "_"), 0.0).reset_index(drop=True)
    df = df.drop(['decoded_towns'], axis=1)
    if 'mispar_ishi' in df:
        df = df.drop(['mispar_ishi'], axis=1)
    return df


def predict_and_train_on_melted_data(train_and_predict_func):
    df = prepare_df()
    y_filter = ['choice_value']
    x_filter = df.columns[~df.columns.isin(y_filter)]
    x_train, x_test, y_train, y_test = split_df_to_validation_and_training(df, y_filter, x_filter)
    print("create classifier on melted df - predict for each role rank between 1-5")
    y_pred, clf, y_train_pred = train_and_predict_func(x_test, x_train, y_train)
    print_results(y_pred, y_test, y_train_pred, y_train, x_train, x_test)
    print("*** done ***")


def train_and_predict_by_xgb_classifier(x_test, x_train, y_train):
    clf = XGBClassifier(silent=False, n_jobs=13, random_state=15, n_estimators=100)
    print('start train model')
    clf = clf.fit(x_train, np.ravel(y_train), eval_metric='rmse')
    print('finish to train model')
    pred = clf.predict(x_test)
    print('finish to predict test set')
    pred_train = clf.predict(x_train)
    print('finish to predict train set')
    return pred, clf, pred_train


if __name__ == '__main__':
    predict_and_train_on_melted_data(train_and_predict_by_xgb_classifier)
