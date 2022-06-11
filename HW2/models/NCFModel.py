from scipy import sparse
from typing import List
import datetime
import os

import lightfm
import numpy as np
import pandas as pd
import tensorflow as tf
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Embedding, Flatten, Concatenate, Multiply, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2

from HW2.add_uniq_featurs.create_features_based_on_kmode_main import get_full_roles_df, add_clusters_to_roles_data, \
    get_full_users_df, train_kmode
from HW2.modles.XGBoost.xgb_classifier_melted_data import manual_category_convert, label_encoding, clean_dataset

TOP_K = 5
N_EPOCHS = 10

def create_ncf(
    number_of_users: int,
    number_of_items: int,
    latent_dim_mf: int = 4,
    latent_dim_mlp: int = 32,
    reg_mf: int = 0,
    reg_mlp: int = 0.01,
    dense_layers: List[int] = [8, 4],
    reg_layers: List[int] = [0.01, 0.01],
    activation_dense: str = "relu",
) -> keras.Model:

    # input layer
    user = Input(shape=(), dtype="int32", name="user_id")
    item = Input(shape=(), dtype="int32", name="item_id")

    # embedding layers
    mf_user_embedding = Embedding(
        input_dim=number_of_users,
        output_dim=latent_dim_mf,
        name="mf_user_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mf),
        input_length=1,
    )
    mf_item_embedding = Embedding(
        input_dim=number_of_items,
        output_dim=latent_dim_mf,
        name="mf_item_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mf),
        input_length=1,
    )

    mlp_user_embedding = Embedding(
        input_dim=number_of_users,
        output_dim=latent_dim_mlp,
        name="mlp_user_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mlp),
        input_length=1,
    )
    mlp_item_embedding = Embedding(
        input_dim=number_of_items,
        output_dim=latent_dim_mlp,
        name="mlp_item_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mlp),
        input_length=1,
    )

    # MF vector
    mf_user_latent = Flatten()(mf_user_embedding(user))
    mf_item_latent = Flatten()(mf_item_embedding(item))
    mf_cat_latent = Multiply()([mf_user_latent, mf_item_latent])

    # MLP vector
    mlp_user_latent = Flatten()(mlp_user_embedding(user))
    mlp_item_latent = Flatten()(mlp_item_embedding(item))
    mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

    mlp_vector = mlp_cat_latent

    # build dense layers for model
    for i in range(len(dense_layers)):
        layer = Dense(
            dense_layers[i],
            activity_regularizer=l2(reg_layers[i]),
            activation=activation_dense,
            name="layer%d" % i,
        )
        mlp_vector = layer(mlp_vector)

    predict_layer = Concatenate()([mf_cat_latent, mlp_vector])

    result = Dense(
        1, activation="sigmoid", kernel_initializer="lecun_uniform", name="interaction"
    )

    output = result(predict_layer)

    model = Model(
        inputs=[user, item],
        outputs=[output],
    )

    return model


def make_tf_dataset(
    df: pd.DataFrame,
    targets: List[str],
    val_split: float = 0.1,
    batch_size: int = 512,
    seed=42,
):
    """Make TensorFlow dataset from Pandas DataFrame.
    :param df: input DataFrame - only contains features and target(s)
    :param targets: list of columns names corresponding to targets
    :param val_split: fraction of the data that should be used for validation
    :param batch_size: batch size for training
    :param seed: random seed for shuffling data - `None` won't shuffle the data"""

    n_val = round(df.shape[0] * val_split)
    if seed:
        # shuffle all the rows
        x = df.sample(frac=1, random_state=seed).to_dict("series")
    else:
        x = df.to_dict("series")
    y = dict()
    for t in targets:
        y[t] = x.pop(t)
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    ds_val = ds.take(n_val).batch(batch_size)
    ds_train = ds.skip(n_val).batch(batch_size)
    return ds_train, ds_val


def prepare_df():
    df = pd.read_csv("../../csv_files/melt_final_manila_data.csv", encoding="UTF-8")
    print("finished to read data")
    df = manual_category_convert(df)
    print("finished to convert category variables")
    df = clean_dataset(df)
    print("finished to clean data")
    label_encoding(df)
    print("df is ready!")
    return df

def wide_to_long(wide: np.array, possible_ratings: List[int]) -> np.array:
    """Go from wide table to long.
    :param wide: wide array with user-item interactions
    :param possible_ratings: list of possible ratings that we may have."""

    def _get_ratings(arr: np.array, rating: int) -> np.array:
        """Generate long array for the rating provided
        :param arr: wide array with user-item interactions
        :param rating: the rating that we are interested"""
        idx = np.where(arr == rating)
        return np.vstack(
            (idx[0], idx[1], np.ones(idx[0].size, dtype="int8") * rating)
        ).T

    long_arrays = []
    for r in possible_ratings:
        long_arrays.append(_get_ratings(wide, r))

    return np.vstack(long_arrays)

if __name__ == '__main__':

    manila_data = pd.read_csv("../csv_files/final_manila_data.csv", encoding="UTF-8")

    sparse_data = manila_data.loc[:,
                  'eshkol diagnosis of manpower_the psychotechnical array':'dedicated track_combat communications officer']
    sparse_data[sparse_data < 3] = -1
    sparse_data[sparse_data >= 3] = 1
    sparse_data.fillna(0, inplace=True)

    n_users = sparse_data.shape[0]
    n_items = sparse_data.shape[1]

    users = manila_data[['mispar_ishi']]

    long_all = wide_to_long(sparse_data, [-1, 0, 1])
    df_all_long = pd.DataFrame(long_all, columns=["user_id", "item_id", "interaction"])

    x_train, x_test, y_train, y_test = train_test_split(df_all_long[["user_id", "item_id"]], df_all_long[["interaction"]],
                                                        test_size=0.2, random_state=1)
    full_train = pd.DataFrame(pd.concat([x_train, y_train], axis=1), columns=["user_id", "item_id", "interaction"])
    full_test = pd.concat([x_train, y_train], axis=1)
    ds_train, ds_val = make_tf_dataset(full_train, ["interaction"])
    ncf_model = create_ncf(n_users, n_items)
    ncf_model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    ncf_model._name = "neural_collaborative_filtering"
    ncf_model.summary()

    train_hist = ncf_model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=N_EPOCHS,
        verbose=1,
    )

    df_test = pd.DataFrame(full_test, columns=["user_id", "item_id", "interaction"])
    ds_test, _ = make_tf_dataset(df_test, ["interaction"], val_split=0, seed=None)
    ncf_predictions = ncf_model.predict(ds_test)
    df_test["ncf_predictions"] = ncf_predictions
    df_test.head()
