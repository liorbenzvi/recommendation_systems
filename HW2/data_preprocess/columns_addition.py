import math
import pandas as pd
import datetime


def get_most_severe_group(row):
    prefix = "seif_likuy"
    max_severity = 0
    max_group = 0
    for i in range(1, 10):
        seif_name = prefix + f"{i:02}"  # adds 0 prefix e.g. "1" -> "01"
        if not math.isnan(row[seif_name]):
            seif_value = str(int(row[seif_name]))
            if seif_value:
                group = int(seif_value[:1])  # Group = left most digit
                severity = int(seif_value[-1:])  # Severity = right most digit
                if severity > max_severity:
                    max_severity = severity
                    max_group = group
    return max_group


def get_units_sum(row):
    prefix = "units"
    sum_units = 0
    for i in range(1, 41):
        unit_name = prefix + f"{i:02}"  # adds 0 prefix e.g. "1" -> "01"
        if not math.isnan(row[unit_name]):
            unit_value = int(row[unit_name])
            sum_units += unit_value
    return sum_units


def get_only_roles_columns(df):
    return df.loc[:, 'eshkol diagnosis of manpower_the psychotechnical array':
                     'martial roles_marine fighting'].columns


def add_cluster_avg_columns(df):
    # Add 2 average columns (with/ without null) for each cluster :
    clusters_names = set(
        get_only_roles_columns(df).map(lambda col: col.split("_")[0]))
    for cluster in clusters_names:
        cluster_col = df.columns[df.columns.str.contains(pat=cluster + "_")]
        df[cluster + "_avg"] = df[cluster_col].mean(axis=1)
        df[cluster + "_avg_null_as_0"] = df[cluster_col].fillna(0).mean(axis=1)


def add_force_avg_columns(df):
    additional_data_on_rolls_df = pd.read_csv("../csv_files/roles_data/additional_data_on_roles.csv",
                                              encoding='latin-1')
    # Add 2 average columns (with/ without null) for each force :
    for force in ["air", "ground", "navy", "general"]:
        force_col = additional_data_on_rolls_df.loc[
            additional_data_on_rolls_df['force'] == force, 'role']
        df[force + "_avg"] = df[force_col].mean(axis=1)
        df[force + "_avg_null_as_0"] = df[force_col].fillna(0).mean(axis=1)


def add_technological_avg_columns(df):
    additional_data_on_rolls_df = pd.read_csv("../csv_files/roles_data/additional_data_on_roles.csv",
                                              encoding='latin-1')
    is_technological_col = additional_data_on_rolls_df.loc[
        additional_data_on_rolls_df['is_technological'] == 'yes', 'role']
    df['is_technical_role_avg'] = df[is_technological_col].mean(axis=1)
    df['is_technical_role_avg_null_as_0'] = df[is_technological_col].fillna(
        0).mean(axis=1)


def add_physical_avg_columns(df):
    additional_data_on_rolls_df = pd.read_csv("../csv_files/roles_data/additional_data_on_roles.csv",
                                              encoding='latin-1')
    is_physical_col = additional_data_on_rolls_df.loc[
        additional_data_on_rolls_df['is_physical'] == 'yes', 'role']
    df['is_physical_role_avg'] = df[is_physical_col].mean(axis=1)
    df['is_physical_role_avg_null_as_0'] = df[is_physical_col].fillna(0).mean(
        axis=1)


def add_leadership_avg_columns(df):
    additional_data_on_rolls_df = pd.read_csv("../csv_files/roles_data/additional_data_on_roles.csv",
                                              encoding='latin-1')
    is_physical_col = additional_data_on_rolls_df.loc[
        additional_data_on_rolls_df['is_leadership'] == 'yes', 'role']
    df['is_leadership_role_avg'] = df[is_physical_col].mean(axis=1)
    df['is_leadership_role_avg_null_as_0'] = df[is_physical_col].fillna(
        0).mean(axis=1)


def get_num_of_likuys(row):
    prefix = "seif_likuy"
    num = 0
    for i in range(1, 10):
        unit_name = prefix + f"{i:02}"  # adds 0 prefix e.g. "1" -> "01"
        if not math.isnan(row[unit_name]):
            num += 1
    return num


def get_max_severity_likuy(row):
    prefix = "seif_likuy"
    max_severity = 0
    for i in range(1, 10):
        unit_name = prefix + f"{i:02}"  # adds 0 prefix e.g. "1" -> "01"
        if not math.isnan(row[unit_name]):
            seif_value = str(int(row[unit_name]))
            severity = int(seif_value[-1:])  # Severity = right most digit
            max_severity = max(severity, max_severity)
    return max_severity


def get_avg_severity_likuy(row):
    prefix = "seif_likuy"
    sum_of_severity = 0
    num_of_likuys = 0
    for i in range(1, 10):
        unit_name = prefix + f"{i:02}"  # adds 0 prefix e.g. "1" -> "01"
        if not math.isnan(row[unit_name]):
            seif_value = str(int(row[unit_name]))
            severity = int(seif_value[-1:])  # Severity = right most digit
            sum_of_severity += severity
            num_of_likuys += 1
    return 0 if num_of_likuys == 0 else sum_of_severity / num_of_likuys


def get_most_comon_likuy(row):
    prefix = "seif_likuy"
    likuys_list = []
    for i in range(1, 10):
        unit_name = prefix + f"{i:02}"  # adds 0 prefix e.g. "1" -> "01"
        if not math.isnan(row[unit_name]):
            seif_value = str(int(row[unit_name]))
            group_size = 2 if len(seif_value) == 5 else 1
            group = int(seif_value[:group_size])
            likuys_list.append(group)
    return 0 if len(likuys_list) == 0 else max(set(likuys_list), key=likuys_list.count)


def add_profess_units(df):
    units_prefix = "units"
    profess_prefix = "mprofesscode"
    for index, row in df.iterrows():
        for i in range(1, 41):
            unit_name = units_prefix + f"{i:02}"  # adds 0 prefix e.g. "1" -> "01"
            profess_name = profess_prefix + f"{i:02}"
            if not math.isnan(row[unit_name]) and row[unit_name] != 0:
                unit_value = int(row[unit_name])
                profess_value = int(row[profess_name])
                df.at[index,
                      "Profess_{}_units".format(profess_value)] = unit_value
            else:
                break


def convert_to_date(date_string):
    date_string_split = date_string.split('-')
    return datetime.datetime(int(date_string_split[0]), int(date_string_split[1]), int(date_string_split[2]))


def get_month_age_when_fill_questionnaire(row):
    t_leida_date = convert_to_date(row['t_leida'])
    manila_answer_date_as_date = convert_to_date(row['manila_answer_date'])
    return ((manila_answer_date_as_date.year - t_leida_date.year) * 12 \
            + (manila_answer_date_as_date.month - t_leida_date.month)) - 202


def month_to_season(month):
    if month <= 3:
        return "winter"
    elif 4 <= month < 7:
        return "spring"
    elif 7 <= month < 10:
        return "summer"
    else:
        return "fall"


def get_born_month(row):
    t_leida_date = convert_to_date(row['t_leida'])
    return t_leida_date.month


def get_fill_month(row):
    manila_answer_date_as_date = convert_to_date(row['manila_answer_date'])
    return manila_answer_date_as_date.month


def get_born_season(row):
    t_leida_date = convert_to_date(row['t_leida'])
    return month_to_season(t_leida_date.month)


def get_fill_season(row):
    manila_answer_date_as_date = convert_to_date(row['manila_answer_date'])
    return month_to_season(manila_answer_date_as_date.month)


def get_decoded_town(row, decode_dict):
    if math.isnan(float(row["yeshuv_code"])):
        return None
    return decode_dict[float(row["yeshuv_code"])]


def add_colloum_from_dict(row, decode_dict):
    if math.isnan(float(row["decoded_towns"])):
        return None
    if float(row["decoded_towns"]) not in decode_dict.keys():
        return None
    return decode_dict[float(row["decoded_towns"])]


def add_columns_to_df(df):
    add_profess_units(df)
    df["most_severe_group"] = df.apply(lambda row: get_most_severe_group(row), axis=1)
    df["sum_of_units"] = df.apply(lambda row: get_units_sum(row), axis=1)
    df["avg_severity"] = df.apply(lambda row: get_avg_severity_likuy(row), axis=1)
    df["max_severity"] = df.apply(lambda row: get_max_severity_likuy(row), axis=1)
    df["most_comon_likuy"] = df.apply(lambda row: get_most_comon_likuy(row), axis=1)
    df["num_of_likuys"] = df.apply(lambda row: get_num_of_likuys(row), axis=1)
    df["age_in_months"] = df.apply(lambda row: get_month_age_when_fill_questionnaire(row), axis=1)
    df["born_in_month"] = df.apply(lambda row: get_born_month(row), axis=1)
    df["fill_in_month"] = df.apply(lambda row: get_fill_month(row), axis=1)
    df["born_in_season"] = df.apply(lambda row: get_born_season(row), axis=1)
    df["fill_in_season"] = df.apply(lambda row: get_fill_season(row), axis=1)
    return df


def add_columns_to_df_for_analysis(df):
    add_cluster_avg_columns(df)
    add_force_avg_columns(df)
    add_technological_avg_columns(df)
    add_physical_avg_columns(df)
    add_leadership_avg_columns(df)
    add_profess_units(df)
    return df


def remove_rows_from_df(df):
    # we don't predict for candidates with adjustment issues
    rows_num_before = len(df.index)
    df = df.drop(df[~df.kahas.isnull()].index)
    rows_num_after = len(df.index)
    print('removed {0} rows'.format(rows_num_before - rows_num_after))
    return df


def remove_columns_from_df(df):
    columns_num_before = len(df.columns)
    df = df.drop(['shnol', 'dapar_hastzurani', 't_leida', 'manila_answer_date', 'yeshuv_code', 'kahas'], axis=1)
    units_prefix = "units"
    profess_prefix = "mprofesscode"
    seif_likuy_prefix = "seif_likuy"

    for i in range(1, 42):
        unit_name = units_prefix + f"{i:02}"  # adds 0 prefix e.g. "1" -> "01"
        profess_name = profess_prefix + f"{i:02}"
        df = df.drop([unit_name], axis=1)
        df = df.drop([profess_name], axis=1)
    for i in range(1, 11):
        seif_likuy_name = seif_likuy_prefix + f"{i:02}"
        df = df.drop([seif_likuy_name], axis=1)

    columns_num_after = len(df.columns)
    print('removed {0} columns'.format(columns_num_before - columns_num_after))
    return df


def add_decoded_towns(df):
    decode = pd.read_csv("../csv_files/users_data/decodedYeshuvs.csv", encoding="UTF-8")
    dic = dict([([a, b]) for a, b in zip(decode.coded_semel, decode.semel)])
    df["decoded_towns"] = df.apply(lambda row: get_decoded_town(row, dic), axis=1)
    return df


def remove_sparse_bagrut_colums(df):
    column_to_number_of_nones = df.notna().sum()
    for column, nones_count in column_to_number_of_nones.iteritems():
        if column.startswith("Profess_") and nones_count < 1380:  # if more than 95% of the values are none - remove it
            df = df.drop([column], axis=1)
    return df


def add_additional_data_from_towns_per_column(df, education, columnName, columnType=None):
    dic = dict([([a, b]) for a, b in zip(education.semel, education[columnName])])
    df[columnName] = df.apply(lambda row: add_colloum_from_dict(row, dic), axis=1)
    if columnType is not None:
        df[columnName] = df[columnName].astype(columnType)
    return df


def add_social_economic_data(df, social_economic, param):
    pass


def add_additional_data_about_towns(df):
    education = pd.read_csv("../csv_files/extarnal_data/education.csv", encoding="UTF-8")
    add_additional_data_from_towns_per_column(df, education, "num_of_schools")
    add_additional_data_from_towns_per_column(df, education, "num_of_class")
    add_additional_data_from_towns_per_column(df, education, "num_of_students")
    add_additional_data_from_towns_per_column(df, education, "avg_num_of_students_in_class")
    add_additional_data_from_towns_per_column(df, education, "drops_out_females")
    add_additional_data_from_towns_per_column(df, education,
                                              "Percentage_of_matriculation_certificate_eligibility_among")
    add_additional_data_from_towns_per_column(df, education, "Percentage_met_universities_threshold")
    add_additional_data_from_towns_per_column(df, education, "Percentage_students")
    add_additional_data_from_towns_per_column(df, education, "Average_students_per_teacher")
    add_additional_data_from_towns_per_column(df, education, "average_weekly_working_hours_per_student")

    social_economic = pd.read_csv("../csv_files/users_data/social_economic_data_by_town.csv")
    add_additional_data_from_towns_per_column(df, social_economic, "town_type", "category")
    add_additional_data_from_towns_per_column(df, social_economic, "town_social_economic_rating")
    return df
