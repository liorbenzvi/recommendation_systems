# this file takes the original data file and transoforms it to our working dataset
import json

import pandas as pd
from googletrans import Translator

hebrew_alphabet = set('אבגדהוזחטיכךלמםנןסעפףצץקרשת')

translator = Translator()

branches = dict()

def enrich(original_df: pd.DataFrame):
    additional_data_df = pd.read_csv('../csv_files/users_data/city_birth_and_fill_date_additional_data.csv', encoding='UTF-8', index_col='mispar_ishi')
    original_df.sort_values(by=['MISPAR_ISHI'], inplace=True)
    additional_data_df.sort_values(by=['mispar_ishi'], inplace=True)
    return pd.concat([original_df, additional_data_df], axis=1)

def is_word_contain_hebrew(some_string):
    for c in hebrew_alphabet:
        if c in some_string:
            return True


def prepare_for_translation(column_name: str):
    column_name = escape_bad_characters(column_name)
    column_name = column_name.replace("משקית", "אחראית")
    column_name = column_name.replace("משגיחת", "אחראית")
    column_name = column_name.replace("יחש", "יחידות שדה")
    return column_name


def translate(original_column_name: str):
    prepared_column_name = prepare_for_translation(original_column_name)

    role = prepared_column_name.split("_")

    branch = role[0]
    profession = role[1]

    if branch in branches:
        branch_translated = branches[branch]
    else:
        branch_translated = translator.translate(branch).text
        branches[branch] = branch_translated

    branch_translated = branch_translated.replace("Cluster", "Eshkol")
    profession = translator.translate(profession).text
    result = branch_translated + "_" + profession
    result = result.lower()
    result = result.replace("kiryat", "controller")
    result = result.replace("again", "shob")
    result = result.replace("bottomity", "staff")
    result = result.replace("easy", "light")
    result = result.replace("wear", "medic")
    result = result.replace("baptism", "marine fighting officer")

    print(original_column_name + " --->  " + result)
    return result


def escape_bad_characters(column_name):
    new_column_name = column_name.replace("\"", "")
    new_column_name = new_column_name.replace("'", "")
    new_column_name = new_column_name.replace("\\", " ")
    new_column_name = new_column_name.replace("/", " ")
    new_column_name = new_column_name.replace(")", " ")
    new_column_name = new_column_name.replace("(", " ")
    return new_column_name


def normalize(column_name: str):
    return column_name.lower()


def try_translate(column_name: str, translation_cache: dict):
    translation_result = translate(column_name)
    translation_cache[column_name] = translation_result
    return translation_result


def translate_and_cache(column_name: str, translation_cache: dict):
    if column_name in translation_cache:
        return translation_cache[column_name]
    elif is_word_contain_hebrew(column_name):
        translation_result = try_translate(column_name, translation_cache)
        return translation_result
    else:
        return column_name


def preprocess_data(df):
    pass


def preprocess(df):
    preprocess_columns(df)
    preprocess_data(df)


def preprocess_columns(df):
    print("------------------ translate columns --------------------")
    translation_cache_file = open("translation.json", "r", encoding='utf8')
    translation_cache = json.load(translation_cache_file)
    df.rename(lambda column_name: translate_and_cache(column_name, translation_cache),
              axis='columns', inplace=True)
    df.rename(lambda column_name: column_name.lower(),axis='columns', inplace=True)
    translation_cache_file.close()

    print("------------------ save translated columns cache in file system --------------------")
    with open('translation.json', 'w', encoding='utf-8') as f:
        json.dump(translation_cache, f, ensure_ascii=False, indent=4)
