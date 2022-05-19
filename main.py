import sys
import pandas as pd
import numpy as np
import random
import time
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from collections import Counter
from kmodes.kmodes import KModes

ratings_file_name = "yelp_data/Yelp_ratings.csv"
yelp_business_file_name = "yelp_data/yelp_business.csv"


# Q1
def load(file_path, year):
    df = pd.read_csv(file_path, encoding="UTF-8")
    return df.loc[df['Year'] == year]


def load_by_list_of_years(file_path, years):
    df = pd.read_csv(file_path, encoding="UTF-8")
    return df.loc[df['Year'].isin(years)]


# Q2
def test_train_split():
    ratings_test_df = load(ratings_file_name, 2015)
    ratings_train_df = load_by_list_of_years(ratings_file_name, [2016, 2017])
    return ratings_test_df, ratings_train_df


# Q3 A
def rmse(predictions, targets):
    return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())


# Q3 B
def accuracy(predictions, targets):
    total = len(predictions)
    correct = len([i for i, j in zip(predictions, targets.values) if i == j])
    return correct / total


# Q4
def train_base_model(k, ratings_train_df, gamma, lambda_param, value):
    train, validate = \
        np.split(ratings_train_df.sample(frac=1, random_state=42),
                 [int(.8 * len(ratings_train_df))])
    target = validate['stars']
    m = train['stars'].mean()
    print('Rating Average is: ' + str(m))
    items_id_map, num_of_items, num_of_users, user_id_map = get_index_maps(train)

    pu = np.random.uniform(low=-1, high=1, size=(num_of_users, k)) * value
    qi = np.random.uniform(low=-1, high=1, size=(k, num_of_items)) * value
    bu = np.full(num_of_users, m)
    bi = np.full(num_of_items, m)

    rmse_old = sys.maxsize
    acc_old = 0
    old_prediction = []
    prediction = []
    old_bi, old_bu, old_pu, old_qi, prediction = create_copy(bi, bu, pu, qi, prediction)
    i = 0
    while True:
        calc_q_p_metrix(bi, bu, gamma, items_id_map, lambda_param, m, pu, qi, train, user_id_map)
        prediction = predict_mf(bi, bu, items_id_map, m, pu, qi, user_id_map, validate)
        rmse_new = rmse(prediction, target)
        acc_new = accuracy(prediction, target)
        print("Iteration number: " + str(i) + ", with RMSE: " + str(rmse_new))
        if rmse_new > rmse_old:
            return old_pu, old_qi, old_bu, old_bi, rmse_old, user_id_map, items_id_map, old_prediction, acc_old
        rmse_old = rmse_new
        acc_old = acc_new
        old_bi, old_bu, old_pu, old_qi, old_prediction = create_copy(bi, bu, pu, qi, prediction)
        i += 1


def get_index_maps(train):
    num_of_users = len(np.unique(train["user_id"]))
    user_id_map = dict(zip(np.unique(train["user_id"]), np.arange(0, num_of_users)))
    num_of_items = len(np.unique(train["business_id"]))
    items_id_map = dict(zip(np.unique(train["business_id"]), np.arange(0, num_of_items)))
    return items_id_map, num_of_items, num_of_users, user_id_map


def predict_mf(bi, bu, items_id_map, m, pu, qi, user_id_map, df):
    prediction = []
    for line in (df[['user_id', 'business_id']]).iterrows():
        curr_user_id, curr_item_id = line[1]
        p = predict_single_user_business_mf(bi, bu, curr_item_id, curr_user_id, items_id_map, m, pu, qi, user_id_map)
        prediction.append(p)
    return prediction


def round_prediction(pred):
    rounded = round(pred)
    if rounded > 5:
        return 5
    if rounded < 1:
        return 1
    return rounded


def predict_single_user_business_mf(bi, bu, curr_item_id, curr_user_id, items_id_map, m, pu, qi, user_id_map):
    user_idx = user_id_map.get(curr_user_id, None)
    item_idx = items_id_map.get(curr_item_id, None)
    if user_idx is None and not (item_idx is None):
        return round_prediction(pu.mean(axis=0).dot(qi[:, item_idx]))
    if not (user_idx is None) and item_idx is None:
        return round_prediction(pu[user_idx].dot(qi.mean(axis=1)))
    if user_idx is None and item_idx is None:
        return round_prediction(m)
    curr_bu = bu[user_idx]
    curr_bi = bi[item_idx]
    return round_prediction(pu[user_idx].dot(qi[:, item_idx]) + curr_bu + curr_bi)


def create_copy(bi, bu, pu, qi, prediction):
    old_pu = pu.copy()
    old_qi = qi.copy()
    old_bu = bu.copy()
    old_bi = bi.copy()
    old_prediction = prediction.copy()
    return old_bi, old_bu, old_pu, old_qi, old_prediction


def calc_q_p_metrix(bi, bu, gamma, items_id_map, lambda_parm, m, pu, qi, train, user_id_map):
    for line in (train[['user_id', 'business_id', 'stars']]).iterrows():
        curr_user_id, curr_item_id, rui = line[1]
        user_idx = user_id_map[curr_user_id]
        item_idx = items_id_map[curr_item_id]
        curr_bu = bu[user_idx]
        curr_bi = bi[item_idx]
        curr_pu = pu[user_idx,]
        curr_qi = qi[:, item_idx]
        eui = rui - m \
              - curr_bi - curr_bu \
              - curr_pu.dot(curr_qi)
        bu[user_idx] = curr_bu + gamma * (eui - lambda_parm * curr_bu)
        bi[item_idx] = curr_bi + gamma * (eui - lambda_parm * curr_bi)
        pu[user_idx] = curr_pu + gamma * (eui * curr_qi - lambda_parm * curr_pu)
        qi[:, item_idx] = curr_qi + gamma * (eui * curr_pu - lambda_parm * curr_qi)


# Q5
def p_q_visualization(p, q):
    # 100 users:
    flat_p_array = p.flatten()[:100]
    x = np.array(range(0, 100))
    plt.scatter(x, flat_p_array)
    plt.title("Plotting P users matrix")
    plt.xlabel("X - index of users")
    plt.ylabel("Y - P flatten values")
    plt.legend()
    plt.show()

    # 100 business:
    flat_q_array = q.flatten()[:100]
    x = np.array(range(0, 100))
    plt.scatter(x, flat_q_array)
    plt.title("Plotting Q business matrix")
    plt.xlabel("X - index of business")
    plt.ylabel("Y - Q flatten values")
    plt.legend()
    plt.show()


def choose_most_common(cat_dict, currRow):
    max = 0
    max_idx = ""
    for i in currRow.split(";"):
        if max < cat_dict.get(i):
            max = cat_dict.get(i)
            max_idx = i
    return max_idx


def read_and_clean_business_df():
    ## read file
    yelp_business_file_df = pd.read_csv(yelp_business_file_name, encoding="UTF-8")
    #yelp_business_file_df = yelp_business_file_df.head(20)
    ## removing business with less then 5 rev
    yelp_business_file_df = yelp_business_file_df[yelp_business_file_df["review_count"] > 5]
    ## remove close business
    yelp_business_file_df = yelp_business_file_df[yelp_business_file_df["is_open"] == 1]
    ## remove field that not providing data to the model
    yelp_business_file_df = yelp_business_file_df.drop(
        ['address', 'latitude', 'longitude', 'is_open'], axis=1)
    ## fill neighborhood nulls with postal_code
    yelp_business_file_df.neighborhood.fillna(yelp_business_file_df.postal_code, inplace=True)
    ## remove other nulls
    yelp_business_file_df = yelp_business_file_df.dropna()

    ## get 30 most common cat and add them has fileds
    list_of_cat = list(yelp_business_file_df["categories"])
    flat_list = []
    for sublist in list_of_cat:
        for item in sublist.split(";"):
            flat_list.append(item)

    counter = Counter(flat_list)
    cat_dict = dict(counter)

    yelp_business_file_df['newCat'] = yelp_business_file_df.apply(
        lambda row: choose_most_common(cat_dict, row['categories']), axis=1)
    yelp_business_file_df = yelp_business_file_df.drop(
        ['categories'], axis=1)

    return yelp_business_file_df.iloc[:, :2], yelp_business_file_df.iloc[:, 2:]


def get_list_of_cat_fileds(yelp_business_file_df):
    cat_idx = []
    i = 0
    for coloum in yelp_business_file_df.columns:
        if coloum != "review_count" and coloum != "score":
            cat_idx.append(i)
        i = i + 1
    return cat_idx


def get_cluster_avg_stars(clusters_rating, cluster):
    return clusters_rating[cluster]


def get_ids_and_avg_stars_df(yelp_business_file_df_name_Id, yelp_business_file_df, clusters):
    yelp_business_file_df_name_Id.insert(0, "cluster", clusters, True)
    yelp_business_file_df_name_Id.insert(0, "stars", yelp_business_file_df['stars'], True)

    clusters_rating = (yelp_business_file_df_name_Id.groupby(['cluster']).mean().to_dict())['stars']
    yelp_business_file_df_name_Id['catStars'] = yelp_business_file_df_name_Id.apply(
        lambda row: get_cluster_avg_stars(clusters_rating, row['cluster']), axis=1)
    yelp_business_ID_and_stars = yelp_business_file_df_name_Id.drop(
        ['name', 'cluster', 'stars'], axis=1)
    return yelp_business_ID_and_stars


# Q6
def train_content_model():

    yelp_business_file_df_name_Id, yelp_business_file_df = read_and_clean_business_df()
    cat_idx = get_list_of_cat_fileds(yelp_business_file_df)


    train, validate = \
        np.split(ratings_train_df.sample(frac=1, random_state=42),
                 [int(.8 * len(ratings_train_df))])
    bestk = 5
    bestrmse = 0
    for x in range(5, 10):
        temp_yelp_business_file_df_name_Id = yelp_business_file_df_name_Id.copy()
        kproto = KPrototypes(n_clusters=x, verbose=2, max_iter=3)
        clusters = kproto.fit_predict(yelp_business_file_df.values, categorical=cat_idx)
        yelp_business_ID_and_stars = get_ids_and_avg_stars_df(temp_yelp_business_file_df_name_Id,
                                                               yelp_business_file_df, clusters)
        yelp_business_ID_and_stars = dict(yelp_business_ID_and_stars.values)
        prediction = predict_content_base(yelp_business_ID_and_stars, validate)
        rmse_new = np.sqrt(((np.array(prediction) - np.array(validate['stars'])) ** 2).mean())
        bestrmse = rmse_new if rmse_new > bestrmse else bestrmse
        bestk = x if rmse_new > bestrmse else bestk
    print("bestrmse: " + str(bestrmse) + ", bestk: " + str(bestk))
    return yelp_business_ID_and_stars


def get_score_for_content_base(id, yelp_business_ID_and_stars, m):
    if id in yelp_business_ID_and_stars.keys():
        return yelp_business_ID_and_stars.get(id)
    else:
        return m


def predict_content_base(yelp_business_ID_and_stars, df):
    m = np.array(list(yelp_business_ID_and_stars.values())).mean()
    prediction = df.apply(lambda row: get_score_for_content_base(row['user_id'], yelp_business_ID_and_stars, m), axis=1)
    return prediction


# Q7
def predict_rating(id_user, id_business, bi, bu, pu, qi, user_id_map, items_id_map, df):
    m = df['stars'].mean()
    mf_pred = predict_single_user_business_mf(bi, bu, id_business, id_user, items_id_map, m, pu, qi, user_id_map)
    content_pred = predict_content_base(yelp_business_ID_and_stars, df)
    return mf_pred, content_pred


# Q8
def compere_models(bi, bu, pu, qi, user_id_map, items_id_map, df,  yelp_business_ID_and_stars):
    mf_predictions = []
    content_predictions = []
    for user, business in df['user_id', 'business_id']:
        mf_pred, content_pred = \
            predict_rating(user, business, bi, bu, pu, qi, user_id_map, items_id_map, df, yelp_business_ID_and_stars)
        mf_predictions.append(mf_pred)
        content_predictions.append(content_pred)

    rmse_mf = rmse(mf_predictions, df['stars'])
    rmse_content = rmse(content_predictions, df['stars'])
    print("RMSE: rmse of mf model: " + str(rmse_mf) + ", rmse of content model: " + str(rmse_content))

    acc_mf = accuracy(mf_predictions, df['stars'])
    acc_content = accuracy(content_predictions, df['stars'])
    print("Accuracy: accuracy of mf model: " + str(acc_mf) + ", accuracy of content model: " + str(acc_content))


if __name__ == '__main__':
    ratings_test_df, ratings_train_df = test_train_split()

    print('Train MF: ')
    bi, bu, pu, qi, rmse, user_id_map, items_id_map, prediction, acc =\
        train_base_model(165, ratings_train_df, 0.015, 0.95, 0.0005)
    print("Final RMSE is: " + str(rmse))
    print("Final accuracy is: " + str(round(acc * 100, 2)) + "%")
    print("Final prediction on validation set histogram: ")
    print({x: prediction.count(x) for x in prediction})
    # p_q_visualization(pu, qi)

    print('Train content model: ')
    yelp_business_ID_and_stars = train_content_model()

    print('Compare models: ')
    compere_models(bi, bu, pu, qi, user_id_map, items_id_map, ratings_test_df,  yelp_business_ID_and_stars)




