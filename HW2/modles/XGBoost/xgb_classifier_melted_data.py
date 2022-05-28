import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


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


def print_values_statistics(y):
    unique, counts = np.unique(y, return_counts=True)
    d = dict(zip(unique, counts))
    sum_values = sum(d.values())
    for i in d.keys():
        print(f'Percentage of {i} ranks in prediction is: {round((d[i] / sum_values) * 100)}%, amount is {d[i]}')
    print('\n\n')


def print_confusion_matrix(y_pred, y_actual):
    unique, counts = np.unique(y_pred, return_counts=True)
    d = dict(zip(unique, counts))
    merged_y = list(zip(list(y_pred), list(y_actual)))
    TOTAL_TP = 0
    TOTAL_FP = 0
    TOTAL_FN = 0
    TOTAL_TN = 0
    TOTAL_COUNT = 0
    for cls in d.keys():
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        count = 0
        for y_pred, y_actual in merged_y:
            pred_val = y_pred == cls
            actual_val = y_actual == cls
            if pred_val == True and actual_val == True:
                TP += 1
                TOTAL_TP += 1
            if pred_val == True and actual_val == False:
                FP += 1
                TOTAL_FP += 1
            if pred_val == False and actual_val == True:
                FN += 1
                TOTAL_FN += 1
            if pred_val == False and actual_val == False:
                TN += 1
                TOTAL_TN += 1
            TOTAL_COUNT += 1
            count += 1
        if count != 0:
            print(f"For class {cls}: "
                  f"TP={TP} ({TP / count * 100:.2f}%), "
                  f"FP={FP} ({FP / count * 100:.2f}%), "
                  f"FN={FN} ({FN / count * 100:.2f}%), "
                  f"TN={TN} ({TN / count * 100:.2f}%)")

    print(f"Confusion Matrix for all classes: "
          f"TP={TOTAL_TP} ({TOTAL_TP / TOTAL_COUNT * 100:.2f}%), "
          f"FP={TOTAL_FP} ({TOTAL_FP / TOTAL_COUNT * 100:.2f}%), "
          f"FN={TOTAL_FN} ({TOTAL_FN / TOTAL_COUNT * 100:.2f}%), "
          f"TN={TOTAL_TN} ({TOTAL_TN / TOTAL_COUNT * 100:.2f}%)")


def print_results(y_pred, y_test, clf, df, x_filter, y_train_pred, y_train):
    total_test = len(y_pred)
    correct_test = len([i for i, j in zip(y_pred, y_test) if i == j])
    total_train = len(y_train_pred)
    train_correct = len([i for i, j in zip(y_train_pred, y_train) if i == j])
    print("Accuracy on Test Set: {0} %".format(str((correct_test / total_test) * 100)))
    print("Accuracy on Train Set: {0} %".format(str((train_correct / total_train) * 100)))
    print("y_pred info :\n")
    df_describe = pd.DataFrame(y_pred)
    print(df_describe.head())
    print_values_statistics(y_pred)
    print_confusion_matrix(y_pred, y_test)


def predict_and_train_on_melted_data(train_and_predict_func):
    df = prepare_df()
    y_filter = ['choice_value']
    x_filter = df.columns[~df.columns.isin(y_filter)]
    x_train, x_test, y_train, y_test = split_df_to_validation_and_training(df, y_filter, x_filter)
    print("create classifier on melted df - predict for each role rank between 1-5")
    y_pred, clf, y_train_pred = train_and_predict_func(x_test, x_train, y_train)
    print_results(y_pred, y_test, clf, df, x_filter, y_train_pred, y_train)
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
