
from HW2 import preprocess, enrich

if __name__ == '__main__':
    print("\n\n\n**************************** read raw data ****************************\n\n\n")
    df = pd.read_csv("../csv_files/manila_data.csv", encoding="UTF-8")

    print("\n\n\n**************************** enrich with extra raw data ****************************\n\n\n")
    df = enrich(df)

    print("\n\n\n**************************** pre-process and clean data ****************************\n\n\n")
    preprocess(df)

    print("\n\n\n**************************** add and remove columns  ****************************\n\n\n")
    df = add_columns_to_df(df)
    df = remove_rows_from_df(df)
    df = add_decoded_towns(df)
    df = add_additional_data_about_towns(df)
    df = remove_columns_from_df(df)
    df = remove_sparse_bagrut_colums(df)

    print("\n\n\n**************************** save final csv file ****************************\n\n\n")
    df.to_csv("../csv_files/final_manila_data.csv")
