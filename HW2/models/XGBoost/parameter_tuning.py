from xgboost import XGBClassifier
from ml_modles.malted_data_base import *
from sklearn.model_selection import RandomizedSearchCV

if __name__ == '__main__':
    df = prepare_df()
    y_filter = ['choice_value']
    x_filter = df.columns[~df.columns.isin(y_filter)]
    x, y = get_x_y(df, x_filter, y_filter)

    param_test = {
        'max_depth': range(3, 80),
        'min_child_weight': range(1, 20),
        'learning_rate': [i / 10.0 for i in range(1, 9)],
        'n_estimators': range(10, 200, 10),
        'gamma': [i / 10.0 for i in range(0, 5)],
        'subsample': [i / 10.0 for i in range(2, 10)],
        'colsample_bytree': [i / 10.0 for i in range(2, 10)]
    }
    gsearch1 = RandomizedSearchCV(estimator=XGBClassifier(), param_distributions=param_test, n_iter=100, cv=3,
                                  verbose=2, random_state=42, n_jobs=-1)
    gsearch1.fit(x, y)
    print('\n\nResults: ')
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)

