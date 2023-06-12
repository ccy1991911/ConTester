from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import torch

srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz", cuda_device=torch.cuda.current_device())

def get_srl_result(one):

    if type(one) == type('str'):
        sent = one
        tokens = TD.utils.my_tokenizer(sent)
        sent, tokens = exchange_nn_with_ving(sent, tokens)
    else:
        tokens = one

    return srl_predictor.predict_tokenized(tokenized_sentence = tokens)


# -----
# get_consequence_and_result
# input: item in srl_result['verbs']
# -----

condition_token_ids = []


def get_locations(tokens, tags, tag, introducer_words):

    locations = []

    l = None
    r = None

    for i in range(0, len(tags)):
        if tags[i] == 'B-%s'%tag and tokens[i].lower() in introducer_words:
            l = i
        if l != None and tags[i] == 'I-%s'%tag:
            r = i
        if tags[i] == 'O' and l != None and r != None:
            locations.append((l, r))
            l = None
            r = None
    if l != None and r != None:
        locations.append((l, r))

    return locations


def get_text_in_range(tokens, l, r):

    text = ''
    for i in range(l, r+1):
        text += ' ' + tokens[i]

    return text.strip()


def get_conditions(sent, item, tag, introducer_words):

    global condition_token_ids

    conditions = []

    tokens = sent.tokens
    tags = item['tags']
    assert len(tags) == len(tokens)

    locations = get_locations(tokens, tags, tag, introducer_words)

    for (l, r) in locations:
        conditions.append((get_text_in_range(tokens, l, r), tokens[l].lower()))
        for i in range(l, r+1):
            condition_token_ids.add(i)

    return conditions


def get_text(sent, item, condition_token_ids):

    tokens = sent.tokens
    tags = item['tags']
    assert len(tags) == len(tokens)

    text = ''
    for i in range(0, len(tags)):
        if tags[i] != 'O' and i not in condition_token_ids:
            text += ' ' + tokens[i]

    return text.strip()


def get_result(sent, item):

    global condition_token_ids

    return get_text(sent, item, condition_token_ids)


def get_condition_and_result(sent, item):

    global condition_token_ids
    condition_token_ids = set()

    conditions = []
    result = None

    dic = {
        'ARGM-TMP': ['when', 'once', 'upon'],
        'ARGM-ADV': ['if'],
        'ARGM-MNR': ['by']
    }

    for tag in dic:
        tmp = get_conditions(sent, item, tag, dic[tag])
        for x in tmp:
            conditions.append(x)

    if len(conditions) > 0:
        result = get_result(sent, item)

    return (conditions, result)


# ------
# get_srl_item_on_level_1
# ------

def get_covered_token_ids(srl_result, k):

    tags = srl_result['verbs'][k]['tags']
    ids = set()
    for i in range(0, len(tags)):
        if tags[i] != 'O':
            ids.add(i)

    return ids


def item_1_covers_item_2(srl_result, item_1, item_2):

    item_1_ids = get_covered_token_ids(srl_result, item_1)
    item_2_ids = get_covered_token_ids(srl_result, item_2)

    flag = True
    for x in item_2_ids:
        if x not in item_1_ids:
            flag = False
            break

    return flag


def no_item_cover_it(srl_result, k):

    flag = True

    for i in range(0, len(srl_result['verbs'])):
        if i == k:
            continue
        if item_1_covers_item_2(srl_result, i, k) == True:
            flag = False
            break

    return flag


def has_subj_or_obj(srl_result, k):

    tags = srl_result['verbs'][k]['tags']
    subj_obj = ['ARG0', 'ARG1']
    for tag in subj_obj:
        if 'B-%s'%tag in tags:
            return True

    return False


def get_conj_word(sent, srl_item_on_level_1):

    tokens = sent.tokens
    covered_ids = set()

    for item in srl_item_on_level_1:
        tags = item['tags']
        for i in range(0, len(tags)):
            if tags[i] != 'O':
                covered_ids.add(i)

    for i in range(len(tokens) - 1, -1, -1):
        if i not in covered_ids and tokens[i].lower() in ['and', 'or', 'but']:
            return tokens[i].lower()

    return 'and'


def get_srl_item_on_level_1(sent, srl_result):

    srl_item_on_level_1 = []
    conj_word = 'and'

    for i in range(0, len(srl_result['verbs'])):
        if no_item_cover_it(srl_result, i) == True and has_subj_or_obj(srl_result, i) == True:
            srl_item_on_level_1.append(srl_result['verbs'][i])

    if len(srl_item_on_level_1) > 1:
        conj_word = get_conj_word(sent, srl_item_on_level_1)

    return (srl_item_on_level_1, conj_word)



