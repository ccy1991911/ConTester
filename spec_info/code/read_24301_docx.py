#!/usr/bin/python
# encoding=utf8

#
# before run this program
# enter the following sql
# 1. delete from paragraph_24301

# 24.301 style name
# ['B1', 'B2', 'B3', 'B4', 'EW', 'EX', 'FP', 'Heading 1', 'Heading 2', 'Heading 3', 'Heading 4', 'Heading 5', 'Heading 6', 'Heading 8', 'NF', 'NO', 'Normal', 'TAN', 'TF', 'TH', 'TT', 'ZA', 'ZB', 'ZT', 'ZU', 'toc 1', 'toc 2', 'toc 3', 'toc 4', 'toc 5', 'toc 6', 'toc 8']

import ccy

import docx
from docx import Document

def is_target_style(name):

    if name.startswith('Heading'):
        return True
    if name == 'Normal':
        return True
    if name.startswith('B'):
        return True

    return False

if __name__ == '__main__':

    sql = 'delete from paragraph_24301'
    ccy.sql_delete(sql)

    docx_path = '../data/3gpp_spec/24301-g80_ccy.docx'

    document = Document(docx_path)

    section = ''
    para_ID = 0
    for para in document.paragraphs:
        if is_target_style(para.style.name):
            para_text = para.text
            para_text = para_text.replace(' ', ' ')
            while '  ' in para_text:
                para_text = para_text.replace('  ', ' ')

            para_style = para.style.name

            if para_style.startswith('Heading'):
                section = para_text.strip().split('\t')[0].strip()

            if len(para_text.strip()) > 0:
                para_ID = para_ID + 1
                sql = 'insert into paragraph_24301 values ("%d", "%s", "%s", "%s")'%(para_ID, ccy.get_sql_string(para_text), para_style, section)
                ccy.sql_insert(sql)

    print('update paragraph_24301: done')
