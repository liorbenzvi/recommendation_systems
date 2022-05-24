import pandas as pd
from collections import Counter
from HW2.data_preprocess.columns_addition import get_only_roles_columns


def add_cluster_labels(df):
    clusters_names = set(get_only_roles_columns(df).map(lambda col: col.split("_")[0]))
    print("Found {0} clusters to predict".format(len(clusters_names)))
    for cluster in clusters_names:
        cluster_col = df.columns[df.columns.str.contains(pat=cluster + "_")]
        df[cluster + "_label"] = (df[cluster_col].fillna(0).mean(axis=1) > 0)
    return df


def remove_nulls_with_roles_cond():
    df = pd.read_csv("../csv_files/final_manila_data.csv", encoding="UTF-8")
    df = add_cluster_labels(df)

    df3 = pd.read_csv("../csv_files/final_manila_data.csv", encoding="UTF-8")
    df3 = df3.fillna(-2)
    df = df.fillna(0)
    df2 = pd.read_csv("../csv_files/all_roles_conditions_csv.csv", encoding="UTF-8")
    outer_list = []
    i = 0
    for role in df2["role"]:
        cluster = role.split("_")[0] + "_label"
        inner_list = []
        df[role][((df["profil"] < df2.loc[df2["role"] == role, "profile"][i]) |
                  (df["dapar"] < df2.loc[df2["role"] == role, "dapar"][i]) |
                  (df["mea_svivat_adraha"] < df2.loc[df2["role"] == role, "mea_svivat_adrah"][i]) |
                  (df["mea_madad_pikud"] < df2.loc[df2["role"] == role, "mea_madad_pikud"][i]) |
                  (df["mea_svivat_ahzaka"] < df2.loc[df2["role"] == role, "mea_svivat_ahzaka"][i]) |
                  (df["mea_svivat_sade"] < df2.loc[df2["role"] == role, "mea_svivat_sade"][i]) |
                  (df["mea_madad_avodat_zevet"] < df2.loc[df2["role"] == role, "mea_madad_avodat_zevet"][i]) |
                  (df["mea_svivat_ibud"] < df2.loc[df2["role"] == role, "mea_svivat_ibud"][i]) |
                  (df["mea_svivat_afaala"] < df2.loc[df2["role"] == role, "mea_svivat_afaala"][i]) |
                  (df["mea_bagrut"] < df2.loc[df2["role"] == role, "mea_bagrut"][i]) |
                  (df["mea_misgeret"] < df2.loc[df2["role"] == role, "mea_misgeret"][i]) |
                  (df["mea_madad_keshev_selectivi"] < df2.loc[df2["role"] == role, "mea_madad_keshev_selectivi"][i]) |
                  (df["mea_madad_keshev_mitmasheh"] < df2.loc[df2["role"] == role, "mea_madad_keshev_mitmasheh"][i]) |
                  (df["mea_svivat_irgun"] < df2.loc[df2["role"] == role, "mea_svivat_irgun"][i]) |
                  (df["mea_madad_hashkaa"] < df2.loc[df2["role"] == role, "mea_madad_hashkaa"][i]) |
                  (df["mea_svivat_tipul"] < df2.loc[df2["role"] == role, "mea_svivat_tipul"][i]))] = -2

        df[role][df[cluster] == 0] = 100

        diff = df[role] - df3[role]
        inner_list.append(role)
        inner_list.append(Counter(diff.values)[0.0])
        inner_list.append(Counter(diff.values)[1.0])
        inner_list.append(Counter(diff.values)[2.0])
        inner_list.append(sum(v for (k, v) in dict(Counter(diff.values)).items() if k < 0))
        inner_list.append(Counter(diff.values)[102])
        inner_list.append(sum(v for (k, v) in dict(Counter(diff.values)).items() if k > 90 and k < 100))
        i = i + 1
        outer_list.append(inner_list)

    statisics = pd.DataFrame(outer_list, columns=['Name', 'no change', 'we_put_null_and_she_add_null',
                                                  'we_didnt_put_null_and_she_add_null', 'we_put_null_and_she_add_value',
                                                  'we_put_null_cluster_right', 'we_put_null_cluster_wrong'])
    statisics.to_csv("../csv_files/statistics_after_using_roles_conditions.csv")

    df.to_csv('../csv_files/final_manila_data_nulls_as_minus1.csv', encoding="utf-8")


if __name__ == '__main__':
    remove_nulls_with_roles_cond()
