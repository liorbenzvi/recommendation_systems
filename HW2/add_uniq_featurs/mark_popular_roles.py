import pandas as pd


def is_popular(row):
    role = row['role']
    amount = final_dict.get(role, 0)
    return amount > 1000


if __name__ == '__main__':
    manila_data = pd.read_csv("../csv_files/melt_final_manila_data.csv", encoding="UTF-8")
    final_dict = {}
    for line in (manila_data[['choice_value', 'role']]).iterrows():
        choice_value, role = line[1]
        if choice_value == 1:
            final_dict[role] = final_dict.get(role, 0) + 1

    roles_data = pd.read_csv("../csv_files/roles_data/additional_data_on_roles.csv", encoding="UTF-8")
    roles_data['is_popular'] = roles_data.apply(lambda r: is_popular(r), axis=1)
    print('Total {0} popular roles'.format(str(roles_data['is_popular'].sum())))
    roles_data.to_csv('../csv_files/roles_data/full_roles_data.csv', encoding="utf-8")

