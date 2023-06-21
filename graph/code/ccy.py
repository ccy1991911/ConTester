import sys
import os

import sqlite3

import spacy
from spacy.lang.en import English

from datetime import datetime

import hashlib

nlp = English()
nlp.add_pipe('sentencizer')

#database_path = '../data/database.db'
database_path = '/home/tangd/chen481/ConfTest/spec_info/data/database.db'


# 3gpp related

def _3gpp_get_para_text_without_format(para_text):

    for i in range(2, 5):
        if len(para_text) >= i and para_text[i-1] == ')':
            para_text = para_text[i:].strip()

    if para_text[0] == '-':
        para_text = para_text[1:].strip()

    if para_text[0].isdigit() and para_text[1] in ['.', '\t']:
        para_text = para_text[2:].strip()

    lst_end_st = ['; and', '; or', ';or', '; and/or', '; and,', '; or,']
    for st in lst_end_st:
        if para_text.endswith(st):
            para_text = para_text[: - (len(st) - 1)]

    return para_text


# time

def print_time():

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# md5

def get_md5(text):

    md5 = hashlib.md5(text.encode('utf-8')).hexdigest()

    return md5




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
