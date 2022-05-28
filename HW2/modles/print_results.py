import numpy as np
import pandas as pd


def print_confusion_matrix(y_pred, y_actual):
    unique, counts = np.unique(y_pred, return_counts=True)
    d = dict(zip(unique, counts))
    merged_y = list(zip(list(y_pred), list(y_actual)))
    TOTAL_TP = 0
    TOTAL_FP = 0
    TOTAL_FN = 0
    TOTAL_TN = 0
    TOTAL_COUNT = 0
    for cls in d.keys():
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        count = 0
        for y_pred, y_actual in merged_y:
            pred_val = y_pred == cls
            actual_val = y_actual == cls
            if pred_val == True and actual_val == True:
                TP += 1
                TOTAL_TP += 1
            if pred_val == True and actual_val == False:
                FP += 1
                TOTAL_FP += 1
            if pred_val == False and actual_val == True:
                FN += 1
                TOTAL_FN += 1
            if pred_val == False and actual_val == False:
                TN += 1
                TOTAL_TN += 1
            TOTAL_COUNT += 1
            count += 1
        if count != 0:
            print(f"For class {cls}: "
                  f"TP={TP} ({TP / count * 100:.2f}%), "
                  f"FP={FP} ({FP / count * 100:.2f}%), "
                  f"FN={FN} ({FN / count * 100:.2f}%), "
                  f"TN={TN} ({TN / count * 100:.2f}%)")

    print(f"Confusion Matrix for all classes: "
          f"TP={TOTAL_TP} ({TOTAL_TP / TOTAL_COUNT * 100:.2f}%), "
          f"FP={TOTAL_FP} ({TOTAL_FP / TOTAL_COUNT * 100:.2f}%), "
          f"FN={TOTAL_FN} ({TOTAL_FN / TOTAL_COUNT * 100:.2f}%), "
          f"TN={TOTAL_TN} ({TOTAL_TN / TOTAL_COUNT * 100:.2f}%)")


def print_values_statistics(y):
    unique, counts = np.unique(y, return_counts=True)
    d = dict(zip(unique, counts))
    sum_values = sum(d.values())
    for i in d.keys():
        print(f'Percentage of {i} ranks in prediction is: {round((d[i] / sum_values) * 100)}%, amount is {d[i]}')
    print('\n\n')


def print_results(y_pred, y_test, y_train_pred, y_train, x_train, x_test):
    print_total_acc(y_pred, y_test, y_train, y_train_pred)
    print_acc_by_class(y_pred, y_test, y_train, y_train_pred)
    print_acc_by_dapar_score(y_pred, y_test, y_train, y_train_pred, x_train, x_test)
    print_acc_by_user_cluster(y_pred, y_test, y_train, y_train_pred, x_train, x_test)
    print_acc_by_role_cluster(y_pred, y_test, y_train, y_train_pred, x_train, x_test)

    rmse = np.sqrt(((np.array(y_pred) - np.array(y_test)) ** 2).mean())
    print("\nRMSE: {0}".format(str(rmse)))

    print("y_pred info :\n")
    df_describe = pd.DataFrame(y_pred)
    print(df_describe.head())
    print_values_statistics(y_pred)
    print_confusion_matrix(y_pred, y_test)


def print_total_acc(y_pred, y_test, y_train, y_train_pred):
    total_test = len(y_pred)
    correct_test = len([i for i, j in zip(y_pred, y_test) if i == j])
    total_train = len(y_train_pred)
    train_correct = len([i for i, j in zip(y_train_pred, y_train) if i == j])
    print("Accuracy on Test Set: {0} %".format(str((correct_test / total_test) * 100)))
    print("Accuracy on Train Set: {0} %".format(str((train_correct / total_train) * 100)))


def print_acc_by_class(y_pred, y_test, y_train, y_train_pred):
    ranks = [100, 0, 1, 2, 3]
    print("Accuracy by ranks:")
    for rank in ranks:
        total_test = len([i for i in y_pred if i == rank])
        correct_test = len([i for i, j in zip(y_pred, y_test) if i == j and i == rank])
        total_train = len([i for i in y_train_pred if i == rank])
        train_correct = len([i for i, j in zip(y_train_pred, y_train) if i == j and i == rank])
        print("Accuracy on Test Set for rank {0}: {1} %".format(str(rank), str((correct_test / total_test) * 100)))
        print("Accuracy on Train Set for rank {0}: {1} %".format(str(rank), str((train_correct / total_train) * 100)))


def print_acc_by_dapar_score(y_pred, y_test, y_train, y_train_pred, x_train, x_test):
    print("Accuracy by dapar score:")
    for d in range(10, 100, 10):
        y_pred_d = []
        y_test_d = []
        for i in range(0, len(y_pred)):
            if x_train["dapar"].iloc(i) == d:
                y_pred_d.append(y_pred[i])
                y_test_d.append(y_test[i])

        y_train_pred_d = []
        y_train_d = []
        for i in range(0, len(y_test)):
            if x_test["dapar"].iloc(i) == d:
                y_train_pred_d.append(y_train_pred[i])
                y_train_d.append(y_train[i])
        print('Accuracy for dapar class: ' + str(d))
        print_total_acc(y_pred_d, y_test_d, y_train_d, y_train_pred_d)


def print_acc_by_user_cluster(y_pred, y_test, y_train, y_train_pred, x_train, x_test):
    print("Accuracy by user cluster:")
    user_data = pd.read_csv('../csv_files/users_data/full_users_data.csv', encoding="UTF-8")
    for c in range(0, 10):
        y_pred_d = []
        y_test_d = []
        for i in range(0, len(y_pred)):
            user = x_train["mispar_ishi"].iloc(i)
            if user_data["cluster"].iloc(user) == c:
                y_pred_d.append(y_pred[i])
                y_test_d.append(y_test[i])

        y_train_pred_d = []
        y_train_d = []
        for i in range(0, len(y_test)):
            user = x_test["mispar_ishi"].iloc(i)
            if user_data["cluster"].iloc(user) == c:
                y_train_pred_d.append(y_train_pred[i])
                y_train_d.append(y_train[i])
        print('Accuracy for user cluster: ' + str(c))
        print_total_acc(y_pred_d, y_test_d, y_train_d, y_train_pred_d)


def print_acc_by_role_cluster(y_pred, y_test, y_train, y_train_pred, x_train, x_test):
    print("Accuracy by role cluster:")
    roles_data = pd.read_csv('../csv_files/roles_data/full_roles_data.csv', encoding="UTF-8")
    for c in range(0, 5):
        y_pred_d = []
        y_test_d = []
        for i in range(0, len(y_pred)):
            role = x_train["role"].iloc(i)
            if roles_data.at[role, 'cluster'] == c:
                y_pred_d.append(y_pred[i])
                y_test_d.append(y_test[i])

        y_train_pred_d = []
        y_train_d = []
        for i in range(0, len(y_test)):
            role = x_test["role"].iloc(i)
            if roles_data.at[role, 'cluster'] == c:
                y_train_pred_d.append(y_train_pred[i])
                y_train_d.append(y_train[i])
        print('Accuracy for role cluster: ' + str(c))
        print_total_acc(y_pred_d, y_test_d, y_train_d, y_train_pred_d)
