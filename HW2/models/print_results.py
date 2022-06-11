import numpy as np
import pandas as pd

ranks = [-1, 0, 1]


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
    print_acc_for_popular_role(y_pred, y_test, y_train, y_train_pred, x_train, x_test)
    business_metrics(y_pred, y_test, y_train, y_train_pred)

    rmse = np.sqrt(((np.array(y_pred) - np.array(y_test)) ** 2).mean())
    print("RMSE: {0}".format(str(rmse)))

    print_diverse_measure_by_class(y_pred, x_test)

    print('\n\n')
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
    print("Accuracy on Test Set: {0} %".format(str(round((correct_test / total_test) * 100, 2))))
    print("Accuracy on Train Set: {0} %".format(str(round((train_correct / total_train) * 100, 2))))


def print_acc_by_class(y_pred, y_test, y_train, y_train_pred):
    print("Accuracy by ranks:")
    for rank in ranks:
        print('Info for rank: ' + str(rank))
        total_test = len([i for i in y_pred if i == rank])
        correct_test = len([i for i, j in zip(y_pred, y_test) if i == j and i == rank])
        total_train = len([i for i in y_train_pred if i == rank])
        train_correct = len([i for i, j in zip(y_train_pred, y_train) if i == j and i == rank])
        if total_test != 0:
            print("Accuracy on Test Set for rank {0}: {1} %".format
                  (str(rank), str(round(correct_test / total_test) * 100, 2)))
        else:
            print('no results in total_test for this rank')
        if total_train != 0:
            print("Accuracy on Train Set for rank {0}: {1} %".format(
                str(rank), str(round((train_correct / total_train) * 100, 2))))
        else:
            print('no results in total_train for this rank')


def print_acc_by_dapar_score(y_pred, y_test, y_train, y_train_pred, x_train, x_test):
    print("Accuracy by dapar score:")
    for d in range(10, 100, 10):
        y_pred_d = []
        y_test_d = []
        for i in range(0, len(y_pred)):
            if x_train["dapar"].values[i] == d:
                y_pred_d.append(y_pred[i])
                y_test_d.append(y_test[i])

        y_train_pred_d = []
        y_train_d = []
        for i in range(0, len(y_test)):
            if x_test["dapar"].values[i] == d:
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
            user = x_train["mispar_ishi"].values[i]
            if user_data["cluster"].values[user-1] == c:
                y_pred_d.append(y_pred[i])
                y_test_d.append(y_test[i])

        y_train_pred_d = []
        y_train_d = []
        for i in range(0, len(y_test)):
            user = x_test["mispar_ishi"].values[i]
            if user_data["cluster"].values[user-1] == c:
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
            role = x_train["role"].values[i]
            if roles_data.at[role, 'cluster'] == c:
                y_pred_d.append(y_pred[i])
                y_test_d.append(y_test[i])

        y_train_pred_d = []
        y_train_d = []
        for i in range(0, len(y_test)):
            role = x_test["role"].values[i]
            if roles_data.at[role, 'cluster'] == c:
                y_train_pred_d.append(y_train_pred[i])
                y_train_d.append(y_train[i])
        print('Accuracy for role cluster: ' + str(c))
        print_total_acc(y_pred_d, y_test_d, y_train_d, y_train_pred_d)


def print_diverse_measure_by_class(y_pred, x_test):
    print("Diverse measure by ranks:")
    for rank in ranks:
        roles_set = set()
        for i in range(0, len(y_pred)):
            if i == rank:
                role = x_test["role"].values[i]
                roles_set.add(role)
        print('For rank {0}, we have {1} different roles'.format(str(rank), str(len(roles_set))))


def print_acc_for_popular_role(y_pred, y_test, y_train, y_train_pred, x_train, x_test):
    print("Accuracy by popular role cluster:")
    roles_data = pd.read_csv('../csv_files/roles_data/full_roles_data.csv', encoding="UTF-8")
    y_pred_d = []
    y_test_d = []
    count_rank_1 = 0
    count_rank_1_and_popular = 0
    for i in range(0, len(y_pred)):
        role = x_train["role"].values[i]
        if y_pred[i] == 1:
            count_rank_1 += 1
        if roles_data.at[role, 'is_popular']:
            y_pred_d.append(y_pred[i])
            y_test_d.append(y_test[i])
            if y_pred[i] == 1:
                count_rank_1_and_popular += 1
    print("Out of {0} prediction with rank 1 on train set we have {1} predictions on popular roles"
          .format(count_rank_1, count_rank_1_and_popular))

    y_train_pred_d = []
    y_train_d = []
    count_rank_1 = 0
    count_rank_1_and_popular = 0
    for i in range(0, len(y_test)):
        if y_pred[i] == 1:
            count_rank_1 += 1
        role = x_test["role"].values[i]
        if roles_data.at[role, 'is_popular']:
            y_train_pred_d.append(y_train_pred[i])
            y_train_d.append(y_train[i])
            if y_pred[i] == 1:
                count_rank_1_and_popular += 1
    print("Out of {0} prediction with rank 1 on test set we have {1} predictions on popular roles"
          .format(count_rank_1, count_rank_1_and_popular))

    print('Accuracy for popular roles: ')
    print_total_acc(y_pred_d, y_test_d, y_train_d, y_train_pred_d)


def business_metrics(y_pred,y_test, y_train_pred, y_train):
    calc_wrong(y_pred, y_test)
    calc_wrong(y_train_pred, y_train)


def calc_wrong(y_pred, y_test):
    total_one_test = len([i for i in y_pred if i == -1])
    total_three_test = len([i for i in y_pred if i == 1])
    one_when_three = len([i for i, j in zip(y_pred, y_test) if i == -1 and j == 1])
    three_when_one = len([i for i, j in zip(y_pred, y_test) if i == 1 and j == -1])
    print("Total -1 in test set is: {0},  " +
          "Total 1 in test set is: {1}. " +
          "Predict -1 when it was 1: {2}, " +
          "Predict 1 when it was -1: {3}. " +
          "Percentage of wrong ones is: {4}%" +
          "Percentage of wrong three is: {5}%" +
          "".format(total_one_test, total_three_test, one_when_three, three_when_one,
                    str(round((one_when_three / total_one_test) * 100, 2)),
                    str(round((three_when_one / total_three_test) * 100, 2))))

