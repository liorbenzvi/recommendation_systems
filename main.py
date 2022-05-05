import pandas as pd
import numpy as np

ratings_file_name = "yelp_data/Yelp_ratings.csv"


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


# Q3
def rmse(predictions, targets):
    return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())


def accuracy(predictions, targets):
    new_predictions = np.concatenate(predictions, axis=0)
    new_targets = np.concatenate(targets, axis=0)
    total = len(new_predictions)
    correct = len([i for i, j in zip(new_predictions, new_targets) if i == j])
    return correct / total


if __name__ == '__main__':
    # business_df = pd.read_csv("yelp_data/yelp_business.csv", encoding="UTF-8")
    # ratings_df = pd.read_csv("yelp_data/Yelp_ratings.csv", encoding="UTF-8")
    # ratings_demo_df = pd.read_csv("yelp_data/Yelp_ratings_DEMO.csv", encoding="UTF-8")
    ratings_test_df, ratings_train_df = test_train_split()
    print(ratings_test_df.head())
    print(ratings_train_df.head())

