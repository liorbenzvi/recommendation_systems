import numpy as np
import pandas as pd
from kmodes.kmodes import KModes

from HW2.data_preprocess.columns_addition import get_only_roles_columns


def get_full_roles_df():
    additional_data_on_roles = pd.read_csv("../csv_files/roles_data/additional_data_on_roles.csv", encoding="UTF-8")
    roles_conditions = pd.read_csv("../csv_files/roles_data/all_roles_conditions_csv.csv", encoding="UTF-8")
    roles_data = additional_data_on_roles.join(roles_conditions.set_index('role'), on='role')
    roles_data['role_cluster'] = roles_data['role'].apply(lambda col: col.split("_")[0])
    return roles_data


def add_clusters_to_roles_data(roles_data):
    train_kmode(roles_data, 5)
    roles_data.to_csv('../csv_files/roles_data/full_roles_data.csv', encoding="utf-8")


def train_kmode(df, k):
    kmode = KModes(n_clusters=k, init="random", n_init=5, verbose=1)
    clusters = kmode.fit_predict(df)
    df.insert(0, "cluster", clusters, True)


def get_full_users_df():
    final_manila_data = pd.read_csv("../csv_files/final_manila_data.csv", encoding="UTF-8")
    final_manila_data = final_manila_data.drop(get_only_roles_columns(final_manila_data), axis=1)
    city_birth_data = pd.read_csv("../csv_files/users_data/city_birth_and_fill_date_additional_data.csv",
                                    encoding="UTF-8")
    users_data = final_manila_data.join(city_birth_data.set_index('mispar_ishi'), on='mispar_ishi')
    users_data = users_data.drop(['t_leida', 'manila_answer_date'], axis=1)
    mapping = {"winter": 0.0, "fall": 1.0, "summer": 2.0, "spring": 3.0}
    users_data['born_in_season'] = users_data['born_in_season'].map(mapping)
    users_data['fill_in_season'] = users_data['fill_in_season'].map(mapping)
    users_data = users_data.replace((np.inf, -np.inf, np.nan, "-", ".."), 0).reset_index(drop=True)
    users_data.Average_students_per_teacher = users_data.Average_students_per_teacher.astype(float)
    users_data.average_weekly_working_hours_per_student = users_data.average_weekly_working_hours_per_student.astype(float)
    users_data = users_data.apply(pd.to_numeric)
    return users_data


if __name__ == '__main__':
    # roles:
    roles_data = get_full_roles_df()
    roles_data = roles_data.fillna(0)
    add_clusters_to_roles_data(roles_data)

    # users:
    users_df = get_full_users_df()
    users_df = users_df.fillna(0)
    train_kmode(users_df, 10)
    users_df.to_csv('../csv_files/users_data/full_users_data.csv', encoding="utf-8")
