import sqlite3

import spacy
from spacy.lang.en import English

nlp = English()
nlp.add_pipe('sentencizer')

database_path = '../data/database.db'

# spacy related

def get_spacy_sents_from_a_para(para):

    doc = nlp(para)

    sents = []
    for sent in doc.sents:
        sents.append(sent.text)

    return sents


# database related

def get_sql_string(st):
    return st.replace('"', '""')


def sql_execute(sql):

    connect = sqlite3.connect(database_path)
    cursor = connect.cursor()

    try:
        cursor.execute(sql)
        connect.commit()
    except Exception as e:
        print(e)

    connect.close()


def sql_insert(sql):

    sql_execute(sql)


def sql_delete(sql):

    sql_execute(sql)


def get_paragraph():

    connect = sqlite3.connect(database_path)
    cursor = connect.cursor()

    dic_para = {}

    try:
        sql = 'select * from paragraph_24301'
        ret = cursor.execute(sql)
        for tmp in ret:
            para_ID = tmp[0]
            para_text = tmp[1]
            para_style = tmp[2]
            para_section = tmp[3]
            dic_para[para_ID] = {'text': para_text, 'style': para_style, 'section': para_section}
    except Exception as e:
        print(e)

    connect.close()

    return dic_para


def get_para_upper_in_format():

    connect = sqlite3.connect(database_path)
    cursor = connect.cursor()

    dic_para_upper_in_format = {}

    try:
        sql = 'select * from paragraph_upper_in_format_24301'
        ret = cursor.execute(sql)
        for tmp in ret:
            para_ID = tmp[0]
            para_ID_upper_in_format = tmp[1]
            dic_para_upper_in_format[para_ID] = para_ID_upper_in_format
    except Exception as e:
        print(e)

    connect.close()

    return dic_para_upper_in_format


def get_para_follow_in_format():

    dic_para_upper_in_format = get_para_upper_in_format()

    dic_para_follow_in_format = {}
    for para_ID in dic_para_upper_in_format:
        para_ID_upper = dic_para_upper_in_format[para_ID]
        if para_ID_upper not in dic_para_follow_in_format:
            dic_para_follow_in_format[para_ID_upper] = []
        dic_para_follow_in_format[para_ID_upper].append(para_ID)

    return dic_para_follow_in_format


def get_concat_sentence():

    connect = sqlite3.connect(database_path)
    cursor = connect.cursor()

    dic_concat_sentence = {}

    try:
        sql = 'select * from concat_sentence_24301'
        ret = cursor.execute(sql)
        for tmp in ret:
            cs_ID = tmp[0]
            para_ID = tmp[1]
            sent_ID = tmp[2]
            cs_text = tmp[3]
            fmt = tmp[4]
            cs_c_text = tmp[5]
            dic_concat_sentence[cs_ID] = {'para_ID': para_ID, 'sent_ID': sent_ID, 'cs_text': cs_text, 'format': fmt, 'cs_c_text': cs_c_text}
    except Exception as e:
        print(e)

    connect.close()

    return dic_concat_sentence


def get_test_purpose_36523():

    connect = sqlite3.connect(database_path)
    cursor = connect.cursor()

    dic_TP = {}

    try:
        sql = 'select * from test_purpose_36523'
        ret = cursor.execute(sql)
        for tmp in ret:
            TP_text = tmp[0]
            with_text = tmp[1]
            when_text = tmp[2]
            then_text = tmp[3]
            dic_TP[TP_text] = {
                'with': with_text,
                'when': when_text,
                'then': then_text
            }
    except Exception as e:
        print(e)

    connect.close()

    return dic_TP


def get_para_ID_from_sent_ID(sent_ID):

    return sent_ID.split(',')[0]
