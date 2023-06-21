import stanza
import NLP.data

#nlp_depparse = stanza.Pipeline(lang = 'en', processors = 'tokenize, mwt, pos, lemma, depparse', tokenize_pretokenized = True)
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


# ---
# get text of a node
# --

def get_text_on_constituency_node_dfs(node, to_delete_node = []):

    if node in to_delete_node:
        return ''

    if len(node.children) == 0:
        return node.label

    text = ''
    for child in node.children:
        text = text + ' ' + get_text_on_constituency_node_dfs(child, to_delete_node)

    return text


def get_text_on_constituency_node(node, to_delete_node = []):

    text = get_text_on_constituency_node_dfs(node, to_delete_node)

    while '  ' in text:
        text = text.replace('  ', ' ')

    if text.endswith(' .'):
        text = text[:-2] + '.'

    return text.strip()


# ---
# get subject
# ---

def get_subject(sent):

    subject_text = ''
    left_text = ''

    root = get_constituency_result(sent.tokens)
    node = root
    while len(node.children) == 1:
        node = node.children[0]
    if len(node.children) >=2 and node.children[0].label == 'NP':
        node = node.children[0]
        subject_text = get_text_on_constituency_node(node)
        left_text = get_text_on_constituency_node(root, [node])

    return (subject_text, left_text)


# ---
# judges
# ---

def is_a_sentence_with_NP_and_VP(sent):

    node = get_constituency_result(sent.tokens)
    while len(node.children) == 1:
        node = node.children[0]
    if len(node.children) == 2:
        child_1 = node.children[0]
        child_2 = node.children[1]
        if child_1.label == 'NP' and child_2.label == 'VP':
            return True

    return False






