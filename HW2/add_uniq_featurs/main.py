import pandas as pd
from kmodes.kmodes import KModes


def get_full_roles_df():
    additional_data_on_roles = pd.read_csv("../csv_files/additional_data_on_roles.csv", encoding="UTF-8")
    roles_conditions = pd.read_csv("../csv_files/all_roles_conditions_csv.csv", encoding="UTF-8")
    roles_data = additional_data_on_roles.join(roles_conditions.set_index('role'), on='role')
    roles_data['role_cluster'] = roles_data['role'].apply(lambda col: col.split("_")[0])
    return roles_data


if __name__ == '__main__':
    roles_data = get_full_roles_df()
    roles_data = roles_data.fillna(0)
    kmode = KModes(n_clusters=5, init="random", n_init=5, verbose=1)
    clusters = kmode.fit_predict(roles_data)
    roles_data.insert(0, "cluster", clusters, True)
    roles_data.to_csv('../csv_files/full_roles_data.csv', encoding="utf-8")





