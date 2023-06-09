import ccy

def get_output_style(style):

    if style.startswith('H'):
        return '[H%s]'%style[-1]

    if style.startswith('B'):
        dic_B = {
            'B1': '\t',
            'B2': '\t\t',
            'B3': '\t\t\t',
            'B4': '\t\t\t\t',
            'B5': '\t\t\t\t\t',
        }
        return '%s[%s]'%(dic_B[style], style)

    if style.startswith('N'):
        return '[Nm]'


dic_para = ccy.get_paragraph()

#file = open('24301.txt', 'w')

for para_ID in dic_para:
    style = dic_para[para_ID]['style']
    text = dic_para[para_ID]['text']

    #file.write('%s {%s} %s\n\n'%(get_output_style(style), para_ID, text))
    print('%s {%s} %s\n\n'%(get_output_style(style), para_ID, text))

#file.close()


