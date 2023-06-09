#!/usr/bin/python
# encoding=utf8

import ccy

dic_para = ccy.get_paragraph()
dic_para_upper = ccy.get_para_upper_in_format()
dic_para_follow = ccy.get_para_follow_in_format()

dic_sent = {}

def get_para_text_without_format(para_ID):

    para_text = dic_para[para_ID]['text'].strip()

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


def get_para_text_without_xiaokuohao(text):

    s = ''
    flag = 0
    for x in text:
        if x == '(':
            flag += 1
            continue
        if x == ')':
            flag -= 1
            continue
        if flag == 0:
            s += x

    while '  ' in s:
        s = s.replace('  ', ' ')

    return s.strip()


def prepare_sent():

    for para_ID in dic_para:

        para_style = dic_para[para_ID]['style']

        if para_style.startswith('Heading'):
            continue

        para_text = get_para_text_without_format(para_ID)
        para_text = get_para_text_without_xiaokuohao(para_text)

        if para_text.startswith('#'):
            para_section = dic_para[para_ID]['section']
            if para_section == '5.5.1.2.5' or para_section == '5.5.1.3.5':
                para_text = 'When the UE receives an ATTACH REJECT message with EMM cause %s'%para_text
            elif para_section == '5.5.2.3.2':
                para_text = 'When the UE receives a DETACH REQUEST message with "re-attach not required" detach type and EMM cause %s'%para_text
            elif para_section == '5.5.3.2.5' or para_section == '5.5.3.3.5':
                para_text = 'When the UE receives a TRACKING AREA UPDATE REJECT message with EMM cause %s'%para_text
            elif para_section == '5.6.1.5':
                para_text = 'When the UE receives a SERVICE REJECT message with EMM cause %s'%para_text

        sents = ccy.get_spacy_sents_from_a_para(para_text)

        dic_sent[para_ID] = {'len': len(sents), 'sent': {}}
        for i in range(0, len(sents)):
            x = sents[i].strip()

            x = x.replace(' ,', ',')
            x = x.replace(' :', ':')
            x = x.replace(' ;', ';')
            x = x.replace(' .', '.')

            dic_sent[para_ID]['sent'][i] = x


def check_in_study_area(para_ID):

    section = dic_para[para_ID]['section']

    lst_section_not_need_to_study = [
        '4.4.2.2', '4.4.2.5', '4.6', '4.8', '4.9',
        '5.3.1.2.2', '5.3.15', '5.3.19',
        '5.5.1.2.5A', '5.5.1.2.5B', '5.5.1.2.5C', '5.5.1.3',
        '5.5.3.2.5A', '5.5.3.2.5B', '5.5.3.3',
        '5.5.4', '5.5.5', '5.5.6',
        '5.6.1.2.1', '5.6.1.2.2', '5.6.1.4.1', '5.6.1.4.2', '5.6.1.5A', '5.6.1.5B',
        '5.6.2.3',
        '6',
    ]

    for s in lst_section_not_need_to_study:
        if section.startswith(s):
            return False

    return True


def is_end_sent(para_ID, sent_text):

    if para_ID not in dic_para_follow:
        return True

    if sent_text[-1] in ['.']:
        return True

    return False


def should_concat_the_following_sent(text):

    if text.endswith(','):
        return True

    if text.endswith(':') and 'follow' not in text.lower():
        return True

    if text[-1].islower() and 'follow' not in text.lower():
        return True

    return False


def concat(text1, text2, para_ID_2):

    if text1.endswith(':'):
        text1 = text1[:-1] + ','

    #return text1 + '\n[+c+] ' + '{%s, %d, %s} '%(para_ID_2, 0, dic_para[para_ID_2]['style']) + text2
    return text1 + ' ' + text2


def get_upper_text(para_ID, i):

    # in paragraph
    if i > 0:
        return get_upper_text(para_ID, i-1) + '\n' + '{%s, %d} '%(para_ID, i) + dic_sent[para_ID]['sent'][i]
    # cross-sentence
    elif i == 0:
        if para_ID in dic_para_upper:
            para_ID_upper = dic_para_upper[para_ID]
            last_sent_text_upper = dic_sent[para_ID_upper]['sent'][dic_sent[para_ID_upper]['len'] - 1]
            first_sent_text = dic_sent[para_ID]['sent'][0]

            if first_sent_text[0].islower():
                if should_concat_the_following_sent(last_sent_text_upper):
                    return concat(get_upper_text(para_ID_upper, dic_sent[para_ID_upper]['len'] - 1), first_sent_text, para_ID)

            return get_upper_text(para_ID_upper, dic_sent[para_ID_upper]['len'] - 1) + '\n' + '{%s, %d} '%(para_ID, i) + dic_sent[para_ID]['sent'][i]

    return '{%s, %d} '%(para_ID, i) + dic_sent[para_ID]['sent'][i]


def main():

    for para_ID in dic_para:

        if check_in_study_area(para_ID) == False:
            continue

        para_style = dic_para[para_ID]['style']
        if para_style.startswith('Heading'):
            continue

        for i in range(0, dic_sent[para_ID]['len']):
            sent_text = dic_sent[para_ID]['sent'][i]
            if is_end_sent(para_ID, sent_text):
                text_with_format = get_upper_text(para_ID, i)
                #file.write('%s\n---\n'%text_with_format)
                print('%s\n---\n'%text_with_format)


if __name__ == '__main__':

    prepare_sent()

    main()

