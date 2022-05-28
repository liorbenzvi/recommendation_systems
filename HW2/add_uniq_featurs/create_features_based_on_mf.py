import pandas as pd
from matrix_factorization import BaselineModel, KernelMF, train_update_test_split
from sklearn.metrics import mean_squared_error


def save_features():
    b_items = matrix_fact.item_biases
    b_items_df = pd.DataFrame(b_items)
    b_items_df.to_csv('../csv_files/roles_data/mf_roles_b.csv', encoding="utf-8")

    qi = matrix_fact.item_features
    qi_df = pd.DataFrame(qi)
    qi_df.to_csv('../csv_files/roles_data/mf_roles_qi.csv', encoding="utf-8")

    b_users = matrix_fact.user_biases
    b_users_df = pd.DataFrame(b_users)
    b_users_df.to_csv('../csv_files/users_data/mf_users_b.csv', encoding="utf-8")

    pu = matrix_fact.user_features
    pu_df = pd.DataFrame(pu)
    pu_df.to_csv('../csv_files/users_data/mf_users_pu.csv', encoding="utf-8")


if __name__ == '__main__':
    manila_data = pd.read_csv("../csv_files/melt_final_manila_data.csv", encoding="UTF-8")
    new_df = manila_data[['mispar_ishi', 'role', 'choice_value']]
    new_df = new_df.rename(columns={"mispar_ishi": "user_id", "role": "item_id", "choice_value": "rating"})
    X = new_df[["user_id", "item_id"]]
    y = new_df["rating"]

    # Prepare data for learning
    (
        X_train_initial,
        y_train_initial,
        X_train_update,
        y_train_update,
        X_test_update,
        y_test_update,
    ) = train_update_test_split(new_df, frac_new_users=0.2)

    # Initial training
    matrix_fact = KernelMF(n_epochs=20, n_factors=100, verbose=1, lr=0.001, reg=0.005)
    matrix_fact.fit(X_train_initial, y_train_initial)

    # Update model with new users
    matrix_fact.update_users(X_train_update, y_train_update, lr=0.001, n_epochs=20, verbose=1)
    pred = matrix_fact.predict(X_test_update)
    rmse = mean_squared_error(y_test_update, pred, squared=False)
    print(f"\nTest RMSE: {rmse:.4f}")

    save_features()






