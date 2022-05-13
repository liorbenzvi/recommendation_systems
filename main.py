import sys
import pandas as pd
import numpy as np
import random

ratings_file_name = "yelp_data/Yelp_ratings_DEMO.csv"


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
    new_predictions = np.concatenate(predictions, axis=0)
    new_targets = np.concatenate(targets, axis=0)
    total = len(new_predictions)
    correct = len([i for i, j in zip(new_predictions, new_targets) if i == j])
    return correct / total


# Q4
def train_base_model(k, ratings_train_df, gamma, lambda_parm):
    train, validate = \
        np.split(ratings_train_df.sample(frac=1, random_state=42),
                 [int(.8 * len(ratings_train_df))])

    m = train['stars'].mean()
    items_id_map, num_of_items, num_of_users, user_id_map = get_index_maps(train)

    pu = np.random.uniform(low=-1, high=1, size=(num_of_users, k)) * 0.005
    qi = np.random.uniform(low=-1, high=1, size=(k, num_of_items)) * 0.005
    ## this is an hyper parmater that used for giving extra info about the user
    bu = np.random.uniform(low=-1, high=1, size=(num_of_users,)) * 0.005
    ## same for items - is it item that usually get high score, or low?
    bi = np.random.uniform(low=-1, high=1, size=(num_of_items,)) * 0.005

    rmse_old = sys.maxsize
    old_bi, old_bu, old_pu, old_qi = create_copy(bi, bu, pu, qi)
    while True:
        calc_q_p_metrix(bi, bu, gamma, items_id_map, lambda_parm, m, pu, qi, train, user_id_map)
        prediction = predict_mf(bi, bu, items_id_map, m, pu, qi, user_id_map, validate)
        rmse_new = rmse(prediction, validate['stars'])
        print("calc rmse: " + str(rmse_new))
        if rmse_new > rmse_old:
            return old_pu, old_qi, old_bu, old_bi, rmse_old, user_id_map, items_id_map
        rmse_old = rmse_new
        old_bi, old_bu, old_pu, old_qi = create_copy(bi, bu, pu, qi)


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


def predict_single_user_business_mf(bi, bu, curr_item_id, curr_user_id, items_id_map, m, pu, qi, user_id_map):
    user_idx = user_id_map.get(curr_user_id, None)
    item_idx = items_id_map.get(curr_item_id, None)
    if user_idx is None and not (item_idx is None):
        return pu.mean(axis=0).dot(qi[:, item_idx])
    if not (user_idx is None) and item_idx is None:
        return pu[user_idx].dot(qi.mean(axis=1))
    if user_idx is None and item_idx is None:
        return m
    curr_bu = bu[user_idx]
    curr_bi = bi[item_idx]
    return pu[user_idx].dot(qi[:, item_idx]) + curr_bu + curr_bi


def create_copy(bi, bu, pu, qi):
    old_pu = pu.copy()
    old_qi = qi.copy()
    old_bu = bu.copy()
    old_bi = bi.copy()
    return old_bi, old_bu, old_pu, old_qi


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


# Q6
def train_content_model(ratings_train_df):
    train, validate = \
        np.split(ratings_train_df.sample(frac=1, random_state=42),
                 [int(.8 * len(ratings_train_df))])
    model = ""
    return model


# Q7
def predict_rating(id_user, id_business, bi, bu, pu, qi, user_id_map, items_id_map, df):
    m = df['stars'].mean()
    mf_pred = predict_single_user_business_mf(bi, bu, id_business, id_user, items_id_map, m, pu, qi, user_id_map)
    content_pred = ""  ## todo - fill
    return mf_pred, content_pred


# Q8
def compere_models(bi, bu, pu, qi, user_id_map, items_id_map, df):
    mf_predictions = []
    content_predictions = []
    for user, business in df['user_id', 'business_id']:
        mf_pred, content_pred = predict_rating(user, business, bi, bu, pu, qi, user_id_map, items_id_map, df)
        mf_predictions.append(mf_pred)
        content_predictions.append(content_pred)

    rmse_mf = rmse(mf_predictions, df['stars'])
    rmse_content = rmse(content_predictions, df['stars'])
    print("RMSE: rmse of mf model: " + str(rmse_mf) + ", rmse of content model: " + str(rmse_content))

    acc_mf = accuracy(mf_predictions, df['stars'])
    acc_content = accuracy(content_predictions, df['stars'])
    print("Accuracy: accuracy of mf model: " + str(acc_mf) + ", accuracy of content model: " + str(acc_content))


if __name__ == '__main__':
    # business_df = pd.read_csv("yelp_data/yelp_business.csv", encoding="UTF-8")
    # ratings_df = pd.read_csv("yelp_data/Yelp_ratings.csv", encoding="UTF-8")
    # ratings_demo_df = pd.read_csv("yelp_data/Yelp_ratings_DEMO.csv", encoding="UTF-8")
    ratings_test_df, ratings_train_df = test_train_split()
    gamma, lambda_parm = 0.03, 0.03
    bi, bu, pu, qi, rmse, user_id_map, items_id_map = train_base_model(150, ratings_train_df, gamma, lambda_parm)
    print("Final RMSE is: " + str(rmse))
