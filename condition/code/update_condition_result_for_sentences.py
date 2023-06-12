#!/usr/bin/python
# encoding=utf8

import os

import sentence

import ccy
import pickle

if __name__ == '__main__':

    dic = {}
    if os.path.exists('../data/condition_result_for_sentences.pickle') == True:
        with open('../data/condition_result_for_sentences.pickle', 'rb') as f:
            dic = pickle.load(f)

    with open('../data/24301_para_n.txt', 'r', encoding = 'utf-8') as f:
        fileContent = f.readlines()

    cnt = 0
    for tmp in fileContent:
        if tmp.startswith('{'):
            sent_text = tmp.strip()
            sent_text = sent_text[sent_text.find('}')+1:].strip()
            md5 = ccy.get_md5(sent_text)
            if md5 not in dic.keys():
                dic[md5] = {
                    'sent_text': sent_text,
                    'mini_tree_set': sentence.analyze_sent(sent_text)
                }
        cnt = cnt + 1
        if cnt%100 == 0:
            print(cnt, '/', len(fileContent))
            ccy.print_time()

    with open('../data/condition_result_for_sentences.pickle', 'wb') as f:
        pickle.dump(dic, f)

    print('Done')
