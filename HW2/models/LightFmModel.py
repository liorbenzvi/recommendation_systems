from scipy import sparse
from typing import List

import numpy as np
import pandas as pd
from lightfm import LightFM
from sklearn.model_selection import train_test_split

from HW2.add_uniq_featurs.create_features_based_on_kmode_main import get_full_roles_df, add_clusters_to_roles_data, \
    get_full_users_df
from HW2.models.print_results import print_results


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


if __name__ == '__main__':
    # roles:
    roles_data = get_full_roles_df()
    roles_data = roles_data.fillna(0)
    add_clusters_to_roles_data(roles_data)
    n_items = roles_data.shape[0]
    # users:
    users_df = get_full_users_df()
    users_df = users_df.fillna(0)
    n_users = users_df.shape[0]
    manila_data = pd.read_csv("../csv_files/final_manila_data.csv", encoding="UTF-8")
    sparse_data = manila_data.loc[:,
                  'eshkol diagnosis of manpower_the psychotechnical array':'dedicated track_combat communications officer']
    sparse_data[sparse_data < 3] = -1
    sparse_data[sparse_data >= 3] = 1
    sparse_data.fillna(0, inplace=True)

    users = manila_data[['mispar_ishi']]

    long_all = wide_to_long(sparse_data, [-1, 0, 1])
    df_all_long = pd.DataFrame(long_all, columns=["user_id", "item_id", "interaction"])

    x_train, x_test, y_train, y_test = train_test_split(df_all_long[["user_id", "item_id"]], df_all_long[["interaction"]],
                                                        test_size=0.2, random_state=1)
    train = pd.concat([x_train, y_train], axis=1)

    lightfm_model = LightFM(loss="warp")
    lightfm_model.fit(sparse.coo_matrix( (y_train["interaction"].array.astype(np.int32), (x_train["user_id"].array.astype(np.int32), x_train["item_id"].array.astype(np.int32)))),
                      epochs=50)

    predictions = lightfm_model.predict(x_test["user_id"].array.astype(np.int32), x_test["item_id"].array.astype(np.int32))
    train_predictions = lightfm_model.predict(x_train["user_id"].array.astype(np.int32), x_train["item_id"].array.astype(np.int32))
    print_results(predictions, y_test, train_predictions, y_train, x_train, x_test)