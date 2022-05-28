import pandas as pd

from HW2.data_preprocess.column_types import ROLES_COLUMNS

if __name__ == '__main__':
    print("**************************** read raw data ****************************")
    df = pd.read_csv("../csv_files/final_manila_data_nulls_as_minus1.csv", encoding="UTF-8")
    df2 = pd.read_csv("../csv_files/roles_data/additional_data_on_roles.csv", encoding="UTF-8")
    id_vars = [column for column in df.columns.values.tolist() if column not in ROLES_COLUMNS and column is not '']
    print("**************************** Melting ****************************")
    df = df.melt(id_vars=id_vars, value_vars=ROLES_COLUMNS, var_name='role', value_name='choice_value')
    print("************************* Matrix Factorization ************************")
    df = pd.merge(df, df2, how='left', on='role')
    df[['cluster', 'role_name']] = df['role'].str.split("_", expand=True)
    print("************************* remove predicate nulls ****************************")
    df.drop(df[df['choice_value'] == -2].index, inplace=True)
    print("************************* Write Final Output ****************************")
    df.to_csv("../csv_files/melt_final_manila_data.csv")
    print("************************* Done ****************************")
    for col in df.columns:
        print(col)