import numpy as np
import string
import json

# ------
# tokenzier
# ------

PPN_list, PUNCT_set = None, None

def load_PPN_PUNCT():
    PPN_file_path = '/home/tangd/chen481/ConfTest/condition/code/TD/PPN_in_24.301.txt'
    PPN_list = list()
    with open(PPN_file_path, 'r', encoding='utf8') as f:
        for line in f:
            cont = line.strip()
            PPN_list.append(cont)
    PPN_list = np.asarray(PPN_list)
    a = np.asarray([len(z) for z in PPN_list])
    ord_a = np.flip(np.argsort(a))
    PPN_list = PPN_list[ord_a]

    PUNCT_set = set(string.punctuation)

    return PPN_list, PUNCT_set


def my_tokenizer(sent):
    global PPN_list, PUNCT_set
    if PPN_list is None or PUNCT_set is None:
        PPN_list, PUNCT_set = load_PPN_PUNCT()

    a = [(0, len(sent))]
    la = [0]
    for ppn in PPN_list:
        b = list()
        lb = list()
        for itvl, lab in zip(a, la):
            if lab == 1:
                b.append(itvl)
                lb.append(1)
                continue
            s, t = itvl[0], itvl[1]
            z = sent.find(ppn, s, t)
            while z >= 0:
                b.append((s, z))
                lb.append(0)
                last_s = 0
                if z + len(ppn) < len(sent) and sent[z + len(ppn)] == 's':
                    last_s = 1
                b.append((z, z + len(ppn) + last_s))
                lb.append(1)
                s = z + len(ppn) + last_s
                z = sent.find(ppn, s, t)
            if s < t:
                b.append((s, t))
                lb.append(0)
        a, la = b, lb

    a = [sent[aa[0]:aa[1]] for aa in a]

    tokens = list()
    for ph, lab in zip(a, la):
        if lab == 1:
            tokens.append(ph)
            continue

        words = ph.split(' ')
        for wd in words:
            i = 0
            while i < len(wd) and wd[i] in PUNCT_set: i += 1
            j = len(wd) - 1
            while j >= i and wd[j] in PUNCT_set: j -= 1
            for z in range(0, i): tokens.append(wd[z])
            if i < j + 1: tokens.append(wd[i:j + 1])
            for z in range(j + 1, len(wd)): tokens.append(wd[z])

    upper_lab = list()
    for token in tokens:
        if (token.isupper() and token != 'I') or (len(token) > 1 and token != 'Is' and token.endswith('s') and token[:-1].isupper()):
            upper_lab.append(1)
        else:
            upper_lab.append(0)

    new_tokens = list()
    i = 0
    while i < len(tokens):
        lab, token = upper_lab[i], tokens[i]
        if lab == 0:
            new_tokens.append(token)
        else:
            _list = [token]
            j = i+1
            while j < len(tokens) and upper_lab[j] == 1:
                _list.append(tokens[j])
                j += 1
            i = j-1
            new_tokens.append(' '.join(_list))
        i += 1
    tokens = new_tokens

    return tokens

# -------
# noun -> verb
# ------

NN_VING_map = None


def load_NN_VING():
    nn_to_ving_file_path = '/home/tangd/chen481/ConfTest/condition/code/TD/noun_to_verbing.json'
    with open(nn_to_ving_file_path) as f:
        NN_VING_map = json.load(f)

    return NN_VING_map


def exchange_nn_with_ving(sent, tokens):
    global NN_VING_map
    if NN_VING_map is None:
        NN_VING_map = load_NN_VING()
    i = 0
    a = list()
    ntokens = list()
    for k, token in enumerate(tokens):
        while sent[i] != token[0]:
            a.append(sent[i])
            i += 1
        i += len(token)
        if token.lower() in NN_VING_map:
            tokens[k] = NN_VING_map[token.lower()]
        a.append(tokens[k])
        ntokens.append(tokens[k])
    nsent = ''.join(a)
    return nsent, ntokens
