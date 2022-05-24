import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse
from html.parser import HTMLParser
import pandas as pd


def get_one(temp_url, role_name):
    list = []
    list.append(role_name)
    response = requests.get(temp_url)
    temp_soup = BeautifulSoup(response.text, "html.parser")
    role_cond = temp_soup.body.find('div', attrs={'class': "font-pauza font-reg margin-bottom-40"}).contents
    i = 0
    for line in role_cond:
        if "מהם תנאי הקבלה?" in line.text or "איך מתקבלים?" in line.text or "תנאי קבלה:" in line.text:
            splited_all_cond = role_cond[i + 2].text.split(",")
            profile = 0
            for cond in splited_all_cond:
                if "פרופיל" in cond:
                    profile = [int(s) for s in cond.split() if s.isdigit()]
                    if (len(profile) > 1):
                        profile = profile[0]
                    elif (len(profile) > 0):
                        profile = profile[0]
            list.append(profile)
            dapar = 0
            for cond in splited_all_cond:
                if "דפ\"ר" in cond:
                    dapar = [int(s) for s in cond.split() if s.isdigit()]
                    if (len(dapar) > 1):
                        dapar = dapar[1]
                    else:
                        dapar = dapar[0]
            list.append(dapar)
        if "ציוני מא\"ה" in line.text:
            splited_all_cond = role_cond[i + 2].text.split(",")
            mea_svivat_adrah = 0
            for cond in splited_all_cond:
                if "הדרכה" in cond:
                    mea_svivat_adrah = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_svivat_adrah)
            mea_madad_pikud = 0
            for cond in splited_all_cond:
                if "פיקוד" in cond:
                    mea_madad_pikud = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_madad_pikud)
            mea_svivat_ahzaka = 0
            for cond in splited_all_cond:
                if "אחזקה טכנית" in cond:
                    mea_svivat_ahzaka = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_svivat_ahzaka)

            mea_svivat_sade = 0
            for cond in splited_all_cond:
                if "שדה" in cond:
                    mea_svivat_sade = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_svivat_sade)
            mea_madad_avodat_zevet = 0
            for cond in splited_all_cond:
                if "עבודת צוות" in cond:
                    mea_madad_avodat_zevet = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_madad_avodat_zevet)

            mea_svivat_ibud = 0
            for cond in splited_all_cond:
                if "עיבוד מידע" in cond:
                    mea_svivat_ibud = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_svivat_ibud)

            mea_svivat_afaala = 0
            for cond in splited_all_cond:
                if "הפעלה טכנית" in cond:
                    mea_svivat_afaala = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_svivat_afaala)

            mea_bagrut = 0
            for cond in splited_all_cond:
                if "בגרות" in cond:
                    mea_bagrut = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_bagrut)

            mea_misgeret = 0
            for cond in splited_all_cond:
                if "מסגרת" in cond:
                    mea_misgeret = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_misgeret)

            mea_madad_keshev_selectivi = 0
            for cond in splited_all_cond:
                if "קשב סלקטיבי" in cond:
                    mea_madad_keshev_selectivi = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_madad_keshev_selectivi)

            mea_madad_keshev_mitmasheh = 0
            for cond in splited_all_cond:
                if "קשב מתמשך" in cond:
                    mea_madad_keshev_mitmasheh = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_madad_keshev_mitmasheh)

            mea_svivat_irgun = 0
            for cond in splited_all_cond:
                if "ארגון" in cond:
                    mea_svivat_irgun = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_svivat_irgun)

            mea_madad_hashkaa = 0
            for cond in splited_all_cond:
                if "השקעה" in cond:
                    mea_madad_hashkaa = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_madad_hashkaa)

            mea_svivat_tipul = 0
            for cond in splited_all_cond:
                if "טיפול" in cond:
                    if len([int(s) for s in cond.split() if s.isdigit()]) > 0:
                        mea_svivat_tipul = [int(s) for s in cond.split() if s.isdigit()][0]
            list.append(mea_svivat_tipul)
        i = i + 1
    while len(list) < 17:
        list.append(0)
    return list

def get_all():
    translation_cache_file = open("translation.json", "r", encoding='utf8')
    translation_cache = json.load(translation_cache_file)
    data = []
    url = "https://www.mitgaisim.idf.il/%D7%9B%D7%AA%D7%91%D7%95%D7%AA/%D7%A8%D7%90%D7%A9%D7%99/%D7%AA%D7%A4%D7%A7%D7%99%D7%93%D7%99%D7%9D/%D7%AA%D7%A4%D7%A7%D7%99%D7%93%D7%99%D7%9D-%D7%9C%D7%91%D7%A0%D7%95%D7%AA/#/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    all_a = soup.findAll('a')
    outer_list = []

    for link in all_a:
        if ('href' in link.attrs and link.attrs['href'].startswith("/תפקידים/")):
            role_name = link.attrs['title']
            role_name = role_name.replace("/", "")
            role_name = role_name.replace("ך", "כ")
            print(role_name)

            temp_url = "https://www.mitgaisim.idf.il" + link.attrs['href']
            list = get_one(temp_url, role_name)

            if len(list) == 17:
                outer_list.append(list)
            else:
                print("do by hand" + role_name)
    return outer_list



if __name__ == '__main__':
    job1 = "קצונה-ייעודית-בחיל-הלוגיסטיקה-מסלול-חץ"
    job = "/תפקידים/"+ job1 + "/"
    url ="https://www.mitgaisim.idf.il" + job
    outer_list = get_all()
    df = pd.DataFrame(outer_list, columns=['Name', 'profile', 'dapar',
                                                   'mea_svivat_adrah',
                                                   'mea_madad_pikud', 'mea_svivat_ahzaka', 'mea_svivat_sade',
                                                   'mea_madad_avodat_zevet', 'mea_svivat_ibud','mea_svivat_afaala',
                                                   'mea_bagrut', 'mea_misgeret',
                                                   'mea_madad_keshev_selectivi', 'mea_madad_keshev_mitmasheh', 'mea_svivat_irgun',
                                                   'mea_madad_hashkaa', 'mea_svivat_tipul'])
    df.to_csv('jobsCond.csv', encoding="utf-8")



