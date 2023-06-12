import stanza
import NLP.data

nlp_depparse = stanza.Pipeline(lang = 'en', processors = 'tokenize, mwt, pos, lemma, depparse', tokenize_pretokenized = True)
nlp_constituency = stanza.Pipeline(lang = 'en', processors = 'tokenize, pos, constituency', tokenize_pretokenized = True)

def get_depparse_result(one):

    if type(one) == type('str'):
        sent = one
        tokens = TD.utils.my_tokenizer(sent)
        sent, tokens = exchange_nn_with_ving(sent, tokens)
    else:
        tokens = one

    doc = nlp_depparse([tokens])
    return doc.sentences[0].words


def get_constituency_result(one):

    if type(one) == type('str'):
        sent = one
        tokens = TD.utils.my_tokenizer(sent)
        sent, tokens = exchange_nn_with_ving(sent, tokens)
    else:
        tokens = one

    doc = nlp_constituency([tokens])
    return doc.sentences[0].constituency
