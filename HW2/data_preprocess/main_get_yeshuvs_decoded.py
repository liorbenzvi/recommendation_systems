from HW2.data_preprocess.columns_addition import *

if __name__ == '__main__':
    age = pd.read_csv("../csv_files/age.csv", encoding="UTF-8")
    gender = pd.read_csv("../csv_files/maleFemale1.csv", encoding="UTF-8")
    leom = pd.read_csv("../csv_files/arabsAndJewish.csv", encoding="UTF-8")
    df = pd.read_csv("../csv_files/more_data_for_students.csv", encoding="UTF-8")


    data = df['YESHUV_CODE'].value_counts()
    df = pd.DataFrame(data)
    df.to_csv("../csv_files/numOfMalshebInYeshuv.csv")


    d = {'semel': 'num'}
    for index1, row1 in age.iterrows():
        for index2, row2 in gender.iterrows():
            if int(row2["semel"]) == row1["semel"]:
                for index3, row3 in leom.iterrows():
                    if int(row2["semel"]) == row3["semel"]:
                        num_of_age618 = row1["age 6-18"]
                        perc_of_women = row2["female"]/ row2["total"]
                        perc_of_jewish = float(row3["jewish"] )/ float(row3["total"])
                        num_of_women_6_18 = num_of_age618 * perc_of_women * perc_of_jewish
                        d[row1["semel"]] = num_of_women_6_18
                        break
                break

    df = pd.DataFrame([d]).T
    df.to_csv('../csv_files/numOfJewishWomenInAge6to18.csv')
