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
    num_of_users = len(np.unique(ratings_train_df["user_id"]))
    num_of_items = len(np.unique(ratings_train_df["business_id"]))


    pu = np.random.uniform(low=-1, high=1, size=(num_of_users,k)) *0.00005
    qi = np.random.uniform(low=-1, high=1, size=(num_of_items,k)) *0.00005
    ## this is an hyper parmater that used for giving extra info about the user
    ## for ex: is this user usually give high scores?  is he memurmar that give lower score for everything?
    bu = np.random.uniform(low=-1, high=1, size=(num_of_users,)) *0.00005
    ## same for items - is it item that usually get high score, or low?
    bi = np.random.uniform(low=-1, high=1, size=(num_of_items,)) *0.00005

    for rui in train['stars']:
        curr_user_id = rui["user_id"]
        curr_item_id = rui["business_id"]
        curr_bu = bu[curr_user_id]
        curr_bi = bi[curr_item_id]
        curr_pu = pu[curr_user_id]
        curr_qi qi[curr_item_id]
        eui = rui - m \
              - curr_bi - curr_bu \
              - curr_pu - curr_qi
        bu = curr_bu + gamma * (eui - lambda_parm * curr_bu)
        bi = curr_bi + gamma * (eui - lambda_parm * curr_bi)
        qi = curr_qi + gamma * (eui * curr_pu - lambda_parm * curr_qi)
        pu = curr_pu + gamma * (eui * curr_qi - lambda_parm * curr_pu)
        res = curr_qi * curr_pu

    ## where do I use K ?
    ### ????
    # rmse_old = sys.maxsize
    # rmse_new = rmse(y_pred, validate['stars'])
    # if (rmse_new < rmse_old):



if __name__ == '__main__':
    # business_df = pd.read_csv("yelp_data/yelp_business.csv", encoding="UTF-8")
    # ratings_df = pd.read_csv("yelp_data/Yelp_ratings.csv", encoding="UTF-8")
    # ratings_demo_df = pd.read_csv("yelp_data/Yelp_ratings_DEMO.csv", encoding="UTF-8")
    ratings_test_df, ratings_train_df = test_train_split()
    gamma, lambda_parm = 0.05, 0.05
    train_base_model(100, ratings_train_df, gamma, lambda_parm)
    print(ratings_test_df.head())
