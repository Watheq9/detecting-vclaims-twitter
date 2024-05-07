# -*- coding: utf-8 -*-
import string
import pandas as pd
import json
import re
import configure
import os


def clean(text):
    text = re.sub(r"http\S+", " ", text)  # remove urls
    text = re.sub(r"RT ", " ", text)  # remove rt
    text = re.sub(r"@[\w]*", " ", text)  # remove handles
    text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text) # remove special characters
    text = re.sub(r"\t", " ", text)  # remove tabs
    text = re.sub(r"\n", " ", text)  # remove line jump
    text = re.sub(r"\s+", " ", text)  # remove extra white space

    # text = remove_harkat(text)
    text = text.strip()
    return text



def remove_punctuations_tashkeel(text):
    """
    The input should be arabic string
    """
    punctuations = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ""" + string.punctuation

    arabic_diacritics = re.compile(
        """
                                ّ    | # Shadda
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatwil/Kashida
                         """,
        re.VERBOSE,
    )

    # remove_punctuations
    translator = str.maketrans("", "", punctuations)
    text = text.translate(translator)

    # remove Tashkeel
    text = re.sub(arabic_diacritics, "", text)

    return text


def remove_longation(text):
    # remove longation
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_harakaat(text):
    # harakaat and tatweel (kashida) to remove
    accents = re.compile(r"[\u064b-\u0652\u0640]")

    # Keep only Arabic letters/do not remove number
    arabic_punc = re.compile(r"[\u0621-\u063A\u0641-\u064A\d+]+")
    text = " ".join(arabic_punc.findall(accents.sub("", text)))
    text = text.strip()
    return text


def normalize(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    return text


def remove_emoji_smileys(text):
    try:
        # UCS-4
        EMOJIS_PATTERN = re.compile(
            u"([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])"
        )
    except re.error:
        # UCS-2
        EMOJIS_PATTERN = re.compile(
            u"([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])"
        )

    SMILEYS_PATTERN = re.compile(r"(\s?:X|:|;|=)(?:-)?(?:\)+|\(|O|D|P|S|\\|\/\s){1,}", re.IGNORECASE)

    text = SMILEYS_PATTERN.sub(r"", text)
    text = EMOJIS_PATTERN.sub(r"", text)
    return text

def remove_punctuation(text):
    # Removing punctuations in string using regex
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_flags(text):
    arabic_punc = re.compile(r"[\u0621-\u063A\u0641-\u064A\d+]+")
    text = " ".join(arabic_punc.findall(text))
    text = text.strip()
    return text


def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+",
        flags=re.UNICODE,
    )

    try:
        # Wide UCS-4 build
        unicode_emoji_pattern = re.compile(
            u"[" u"\U0001F300-\U0001F64F" u"\U0001F680-\U0001F6FF" u"\u2600-\u26FF\u2700-\u27BF]+",
            re.UNICODE,
        )
    except re.error:
        # Narrow UCS-2 build
        unicode_emoji_pattern = re.compile(
            u"("
            u"\ud83c[\udf00-\udfff]|"
            u"\ud83d[\udc00-\ude4f\ude80-\udeff]|"
            u"[\u2600-\u26FF\u2700-\u27BF])+",
            re.UNICODE,
        )

    text = unicode_emoji_pattern.sub(u"", text)
    text = emoji_pattern.sub(r"", text)

    return text


def remove_stop_words(text, stop_words):
    # Arabic stop words with nltk
    from nltk.corpus import stopwords

    stop_words = stopwords.words()
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


def clean_and_normalized_text(text):
    # text = text.replace('"', ' ')  # remove double quotes
    text = normalize(text)
    text = clean(text)
    return text


def clean_query(query, lang="ar"):
    query = clean(query)
    query = remove_emoji_smileys(query)

    if lang == "ar":
        query = normalize(query)

        query =  remove_punctuations_tashkeel(query)

        if configure.REMOVE_FLAGS:
            query =  remove_harakaat(query)

        query =  clean(query)

    return query


def get_json(file_path):
    data = -1
    with open(file_path, encoding="utf-8") as data_file:
        data = json.load(data_file)
    return data


def read_excel_file(file_name, sheet_name="Sheet1"):
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    return df


def read_tsv_file(file_name, seperator="\t"):
    df = pd.read_csv(file_name, sep=seperator)
    return df


def save_to_excel(df, file_name, sheet_name="Sheet1"):
    df.to_excel(file_name, sheet_name=sheet_name)


def count_unique_common_words(list1, list2):
    list3 = set(list1) & set(list2)
    common_words = sorted(list3, key=lambda k: list1.index(k))
    # print(len(common_words))
    return len(common_words)


def count_common_words(list1, list2):
    list3 = set(list1)
    tot_cnt = 0
    for word in list3:
        cnt1 = list1.count(word)
        cnt2 = list2.count(word)
        tot_cnt = tot_cnt + min(cnt1, cnt2)
    return tot_cnt


def count_arbitrary_common_words(list1, list2):
    return len([w for w in list2 if w in list1])



# read file based on its extension (tsv or xlsx)
def read_file(input_file, sep="\t", names = ""):
    if input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    else:
        if names != "":
            df = pd.read_csv(input_file, sep=sep, names=names,encoding="utf-8")
        else:
            df = pd.read_csv(input_file, sep=sep,encoding="utf-8")
    return df





def put_all_vclaims_in_one_file(vclaims_dir, vlcaims_file_save_path):
    df = pd.DataFrame()
    for filename in os.listdir(vclaims_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(vclaims_dir, filename)
            json_obj = get_json(file_path)
            df_one_row = pd.json_normalize(json_obj)
            df = df.append(df_one_row, ignore_index=True)

    df.to_excel(vlcaims_file_save_path, index=False)

    

