import ccy

dic_para = ccy.get_paragraph()

cnt = 0
for para_ID in dic_para:
    para_text = dic_para[para_ID]['text']
    sentences = ccy.get_spacy_sents_from_a_para(para_text)
    cnt += len(sentences)

print(cnt)
