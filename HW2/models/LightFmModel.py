from scipy import sparse
from typing import List

import numpy as np
import pandas as pd
from lightfm import LightFM
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from HW2.add_uniq_featurs.create_features_based_on_kmode_main import get_full_roles_df, add_clusters_to_roles_data, \
    get_full_users_df
from HW2.models.print_results import print_results


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


def clean_dataset(df):
    df = df.replace((np.inf, -np.inf, np.nan, "-", "..", "_"), 0.0).reset_index(drop=True)
    df = df.drop(['decoded_towns'], axis=1)
    if 'mispar_ishi' in df:
        df = df.drop(['mispar_ishi'], axis=1)
    return df


def label_encoding(df):
    le = preprocessing.LabelEncoder()
    cols_for_transform = ['role', 'cluster', 'role_name']
    for col in cols_for_transform:
        df[col] = le.fit_transform(df[col])


def manual_category_convert(df):
    df = convert_category_variables(df)
    mapping = {"general": 0.0, "air": 1.0, "navy": 2.0, "ground": 3.0}
    df['force'] = df['force'].map(mapping)
    text_cols = ['is_technological', 'is_physical', 'is_leadership']
    txt_mapping = {"no": 0.0, "yes": 1.0}
    for col in text_cols:
        df[col] = df[col].map(txt_mapping)
    return df


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


def wide_to_long(wide: np.array, possible_ratings: List[int]) -> np.array:
    """Go from wide table to long.
    :param wide: wide array with user-item interactions
    :param possible_ratings: list of possible ratings that we may have."""

    def _get_ratings(arr: np.array, rating: int) -> np.array:
        """Generate long array for the rating provided
        :param arr: wide array with user-item interactions
        :param rating: the rating that we are interested"""
        idx = np.where(arr == rating)
        return np.vstack(
            (idx[0], idx[1], np.ones(idx[0].size, dtype="int8") * rating)
        ).T

    long_arrays = []
    for r in possible_ratings:
        long_arrays.append(_get_ratings(wide, r))

    return np.vstack(long_arrays)


def single_round_prediction(singel_pred):
    rounded = np.round(singel_pred)
    if rounded > 3:
        return 3
    if rounded < 0:
        return 0
    return rounded


def round_prediction(pred):
    return [single_round_prediction(x) for x in pred]


def userid_to_mispar_ishi(userid, userid_to_mispar_ishi_dic):
    userid_to_mispar_ishi_dic = userid_to_mispar_ishi_dic.to_dict()["mispar_ishi"]
    return [userid_to_mispar_ishi_dic.get(x) for x in np.array(userid.array)]


def itemid_to_item_name(items_id, items_names_array):
    return [items_names_array[x] for x in np.array(items_id.array, dtype=int)]


def get_dapar(mispar_ishi, manila_data, extra_data_col_name):
    return manila_data[extra_data_col_name].iloc[mispar_ishi]


def get_extra_data_df(x,manila_data,users,items):
    mispar_ishi = userid_to_mispar_ishi(x["user_id"], users)
    dapar = get_dapar(x["user_id"], manila_data, "dapar")
    role = itemid_to_item_name(x["item_id"], items)
    x_ext = pd.DataFrame()
    x_ext["mispar_ishi"] = mispar_ishi
    x_ext["dapar"] = dapar.values
    x_ext["role"] = role
    return x_ext



if __name__ == '__main__':
    print('Prepare roles data')
    roles_data = pd.read_csv("../csv_files/roles_data/full_roles_data.csv", encoding="UTF-8")
    roles_data = roles_data.fillna(0)
    n_items = roles_data.shape[0]

    print('Prepare users data')
    users_df = get_full_users_df()
    users_df = users_df.fillna(0)
    n_users = users_df.shape[0]

    print('Prepare sparse data')
    manila_data = pd.read_csv("../csv_files/final_manila_data.csv", encoding="UTF-8")
    sparse_data = manila_data.loc[:,
                  'eshkol diagnosis of manpower_the psychotechnical array':'dedicated track_combat communications officer']
    sparse_data[sparse_data < 3] = 1
    sparse_data[sparse_data == 3] = 2
    sparse_data[sparse_data > 3] = 3
    sparse_data.fillna(0, inplace=True)
    users = manila_data[['mispar_ishi']]
    items = sparse_data.columns.values
    long_all = wide_to_long(sparse_data, [0, 1, 2, 3])
    df_all_long = pd.DataFrame(long_all, columns=["user_id", "item_id", "interaction"])

    print('Split to train and test')
    x_train, x_test, y_train, y_test = train_test_split(df_all_long[["user_id", "item_id"]],
                                                        df_all_long[["interaction"]],
                                                        test_size=0.2, random_state=1)

    x_train_ext = get_extra_data_df(x_train,manila_data,users,items)
    x_test_ext = get_extra_data_df(x_test,manila_data,users,items)
    train = pd.concat([x_train, y_train], axis=1)

    print('Train lightFM')
    lightfm_model = LightFM(loss="warp")
    lightfm_model.fit(sparse.coo_matrix((y_train["interaction"].array.astype(np.int32),
                                         (x_train["user_id"].array.astype(np.int32),
                                          x_train["item_id"].array.astype(np.int32)))), epochs=2000)

    print('Predict Test')
    predictions = round_prediction(lightfm_model.predict(x_test["user_id"].array.astype(np.int32),
                                                         x_test["item_id"].array.astype(np.int32)))

    print('Predict Train')
    train_predictions = round_prediction(lightfm_model.predict(x_train["user_id"].array.astype(np.int32),
                                                               x_train["item_id"].array.astype(np.int32)))

    print_results(predictions, y_test.values, train_predictions, y_train.values, x_train_ext, x_test_ext)
