import ccy

dic_para = ccy.get_paragraph()


def get_para_ID_upper(para_ID):

    para_style = dic_para[para_ID]['style']
    para_text = dic_para[para_ID]['text']

    if para_style.startswith('Heading'):
        return None

    if para_style == 'Normal':
        return None

    for i in range(int(para_ID) - 1, 0, -1):
        para_ID_upper = str(i)
        para_style_upper = dic_para[para_ID_upper]['style']
        para_text_upper = dic_para[para_ID_upper]['text']

        if para_style_upper.startswith('Heading'):
            return None

        if para_style_upper == 'Normal':
            return para_ID_upper

        if para_style_upper.startswith('B'):
            if para_style_upper < para_style:
                return para_ID_upper
            if para_text.startswith('\t') == True:
                if para_text_upper.startswith('\t') == False:
                    if para_style_upper == para_style:
                        return para_ID_upper

    return None


if __name__ == '__main__':

    sql = 'delete from paragraph_upper_in_format_24301'
    ccy.sql_delete(sql)

    for para_ID in dic_para:

        para_ID_upper = get_para_ID_upper(para_ID)

        if para_ID_upper != None:
            sql = 'insert into paragraph_upper_in_format_24301 values ("%s", "%s")'%(para_ID, para_ID_upper)
            ccy.sql_insert(sql)

    print('update paragraph_upper_in_format_24301: done')

