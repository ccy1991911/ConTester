import os
import sys

from TD.CosineMimicLoss import CosineMimicLoss
from TD.CosineMimicLoss import myEvaluator
from TD.CosineMimicLoss import get_callback_save_fn

import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

import os
import math
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from sentence_transformers import SentenceTransformer, InputExample, models
from typing import Iterable, Dict
from sentence_transformers.util import batch_to_device

import ccy

import queue


messages_network_to_UE = [
    'ATTACH ACCEPT',
    'ATTACH REJECT',
    'AUTHENTICATION REJECT',
    'AUTHENTICATION REQUEST',
    'DETACH ACCEPT',
    'DETACH REQUEST',
    'DOWNLINK NAS TRANSPORT',
    'EMM INFORMATION',
    'GUTI REALLOCATION COMMAND',
    'IDENTITY REQUEST',
    'SECURITY MODE COMMAND',
    'SERVICE REJECT',
    'TRACKING AREA UPDATE ACCEPT',
    'TRACKING AREA UPDATE REJECT',
    'DOWNLINK GENERIC NAS TRANSPORT',
    'SERVICE ACCEPT'
]

message_UE_to_network = [
    'ATTACH COMPLETE',
    'ATTACH REQUEST',
    'AUTHENTICATION FAILURE',
    'AUTHENTICATION RESPONSE',
    'DETACH ACCEPT',
    'DETACH REQUEST',
    'EXTENDED SERVICE REQUEST',
    'GUTI REALLOCATION COMPLETE',
    'IDENTITY RESPONSE',
    'SECURITY MODE COMPLETE',
    'SECURITY MODE REJECT',
    'SERVICE REQUEST',
    'TRACKING AREA UPDATE COMPLETE',
    'TRACKING AREA UPDATE REQUEST',
    'UPLINK NAS TRANSPORT',
    'UPLINK GENERIC NAS TRANSPORT',
    'CONTROL PLANE SERVICE REQUEST'
]


class Node():
    def __init__(self, ID, weight, cat, condition_or_result, content_text, sent_text, start_Node = False, end_Node = False):

        self.ID = ID
        self.weight = weight
        self.cat = cat
        self.condition_or_result = condition_or_result
        self.content_text = content_text
        self.sent_text = sent_text
        self.start_Node = start_Node
        self.end_Node = end_Node

class Edge():
    def __init__(self, ID, weight, in_Node_ID, out_Node_ID):

        self.ID = ID
        self.weight = weight
        self.in_Node_ID = in_Node_ID
        self.out_Node_ID = out_Node_ID

EDG_edge = {}
EDG_node = {}
dic_text_2_embeddings = {}

device='cuda'
model = SentenceTransformer('../data/model_part1.pt', device=device)
train_loss = torch.load('../data/model_part2.pt', map_location = device)
train_loss.model = model
train_loss = train_loss.to(device)
train_loss.set_predict()
train_loss.eval()

def preparation():

    global EDG_edge
    global EDG_node
    global dic_text_2_embedding

    with open('../data/EDG_graph_node.pkl', 'rb') as f:
        EDG_node = pickle.load(f)
    with open('../data/EDG_graph_edge.pkl', 'rb') as f:
        EDG_edge = pickle.load(f)
    with open('../data/EDG_embeddings.pkl', 'rb') as f:
        dic_text_2_embedding = pickle.load(f)

    print('Done: preparation')

    return


def get_embeddings(text_list):

    new_text_list = list()
    for text in text_list:
        if text not in dic_text_2_embedding:
            new_text_list.append(text)

    print('new text:', len(new_text_list))

    embeddings = model.encode(new_text_list, batch_size = 1024, show_progress_bar = True)
    emb_dict = {sent: emb for sent, emb in zip(new_text_list, embeddings)}

    for text in new_text_list:
        dic_text_2_embedding[text] = emb_dict[text]

    with open('../data/EDG_embeddings.pkl', 'wb') as f:
        pickle.dump(dic_text_2_embedding, f)

    print('Done: update embeddings')

    return


def build_start_Node_set():

    start_Node_symantic = []
    for m in messages_network_to_UE:
        start_Node_symantic.append('the UE receives a %s message'%m)
        start_Node_symantic.append('%s message is received'%m)
        start_Node_symantic.append('receive a %s message from the MME'%m)
        start_Node_symantic.append('the MME sends a %s message'%m)
        start_Node_symantic.append('%s message is sent'%m)
        start_Node_symantic.append('send a %s message to the UE'%m)


    get_embeddings(start_Node_symantic)

    def get_predict_pair():

        to_predict_pair = []

        for node_ID in EDG_node:
            if EDG_node[node_ID].cat == 'text':
                for start_message in start_Node_symantic:
                    to_predict_pair.append((node_ID, start_message))

        return to_predict_pair


    def get_compare_result_md52result(from_pickle = False):

        def check_predict_and_probability(all_preds, all_probs, i):

            threshold_1 = 0.69
            threshold_23 = 0.9

            if all_preds[i] == 1 and all_preds[i+1] == 1:
                if all_probs[i][1] >= threshold_1 and all_probs[i+1][1] >= threshold_1:
                    return True

            if all_preds[i] == 2 and all_preds[i+1] == 3:
                if all_probs[i][2] >= threshold_23 and all_probs[i+1][3] >= threshold_23:
                    return True

            if all_preds[i] == 3 and all_preds[i+1] == 2:
                if all_probs[i][3] >= threshold_23 and all_probs[i+1][2] >= threshold_23:
                    return True

            return False

        compare_result_md52result = {}
        if from_pickle == True:
            with open('../data/EDG_match_result.pkl', 'rb') as f:
                compare_result_md52result = pickle.load(f)
                print('previous record md5 cnt:', len(compare_result_md52result))

        to_predict_pair_text_md52text = {}
        for node_ID, text2 in to_predict_pair:
            text1 = EDG_node[node_ID].content_text
            md5 = ccy.get_md5(text1+text2)
            if md5 not in to_predict_pair_text_md52text and md5 not in compare_result_md52result:
                to_predict_pair_text_md52text[md5] = (text1, text2)
        print('[NEW] to predict pair text:', len(to_predict_pair_text_md52text))

        id_2_text_pair_md5 = {}
        cnt = 0
        all_input = list()
        all_label = list()
        for md5 in to_predict_pair_text_md52text:
            sent_1_text, sent_2_text = to_predict_pair_text_md52text[md5]

            id_2_text_pair_md5[cnt] = md5
            cnt += 1
            id_2_text_pair_md5[cnt] = md5
            cnt += 1

            emb0, emb1 = dic_text_2_embedding[sent_1_text], dic_text_2_embedding[sent_2_text]
            cated_emb = np.concatenate([emb0,emb1], axis=-1)
            all_input.append(cated_emb)
            all_label.append(0)
            cated_emb = np.concatenate([emb1, emb0], axis=-1)
            all_input.append(cated_emb)
            all_label.append(0)

        print('Done: load to input to be predicted')

        if len(all_input) > 0:
            all_input = np.asarray(all_input)
            all_label = np.asarray(all_label)

            input_tensor = torch.from_numpy(all_input)
            label_tensor = torch.from_numpy(all_label)
            train_dataset = TensorDataset(input_tensor, label_tensor)
            train_dataloader = DataLoader(train_dataset, shuffle = False, batch_size = 4096)

            classifier_model = torch.nn.Sequential(OrderedDict(
                [('embedding', train_loss.embedding),
                ('classifier', train_loss.classifier),
                ]
            )).to(device)

            all_probs = list()
            all_preds = list()
            for inputs, labels in train_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = classifier_model(inputs)

                probs = torch.nn.functional.softmax(logits, -1)
                predict = torch.argmax(logits, dim = -1)

                all_probs.append(probs.detach().cpu().numpy())
                all_preds.append(predict.detach().cpu().numpy())

            all_probs = np.concatenate(all_probs,0)
            all_preds = np.concatenate(all_preds,0)

            print('Done: finish predict')

            for i in range(0, len(all_preds), 2):
                md5 = id_2_text_pair_md5[i]
                compare_result_md52result[md5] = False
                if check_predict_and_probability(all_preds, all_probs, i) == True:
                    compare_result_md52result[md5] = True

        with open('../data/EDG_match_result.pkl', 'wb') as f:
            pickle.dump(compare_result_md52result, f)

        match_cnt = 0
        for md5 in compare_result_md52result:
            if compare_result_md52result[md5] == True:
                match_cnt += 1
        print('match cnt total:', match_cnt)

        return compare_result_md52result

    to_predict_pair = get_predict_pair()
    compare_result_md52result = get_compare_result_md52result(True)

    for node_ID, text2 in to_predict_pair:
        text1 = EDG_node[node_ID].content_text
        md5 = ccy.get_md5(text1 + text2)
        if compare_result_md52result[md5] == True:
            EDG_node[node_ID].start_Node = True

    with open('../data/EDG_graph_node.pkl', 'wb') as f:
        pickle.dump(EDG_node, f)

    return

def build_end_Node_set():

    end_Node_symantic = []
    for m in message_UE_to_network:
        end_Node_symantic.append('the UE receives a %s message'%m)
        end_Node_symantic.append('%s message is received'%m)
        end_Node_symantic.append('receive a %s message from the MME'%m)
        end_Node_symantic.append('the MME sends a %s message'%m)
        end_Node_symantic.append('%s message is sent'%m)
        end_Node_symantic.append('send a %s message to the UE'%m)

    get_embeddings(end_Node_symantic)

    def get_predict_pair():

        to_predict_pair = []

        for node_ID in EDG_node:
            if EDG_node[node_ID].cat == 'text':
                for start_message in end_Node_symantic:
                    to_predict_pair.append((node_ID, start_message))

        return to_predict_pair


    def get_compare_result_md52result(from_pickle = False):

        def check_predict_and_probability(all_preds, all_probs, i):

            threshold_1 = 0.69
            threshold_23 = 0.9

            if all_preds[i] == 1 and all_preds[i+1] == 1:
                if all_probs[i][1] >= threshold_1 and all_probs[i+1][1] >= threshold_1:
                    return True

            if all_preds[i] == 2 and all_preds[i+1] == 3:
                if all_probs[i][2] >= threshold_23 and all_probs[i+1][3] >= threshold_23:
                    return True

            if all_preds[i] == 3 and all_preds[i+1] == 2:
                if all_probs[i][3] >= threshold_23 and all_probs[i+1][2] >= threshold_23:
                    return True

            return False

        compare_result_md52result = {}
        if from_pickle == True:
            with open('../data/EDG_match_result.pkl', 'rb') as f:
                compare_result_md52result = pickle.load(f)
                print('previous record md5 cnt:', len(compare_result_md52result))

        to_predict_pair_text_md52text = {}
        for node_ID, text2 in to_predict_pair:
            text1 = EDG_node[node_ID].content_text
            md5 = ccy.get_md5(text1+text2)
            if md5 not in to_predict_pair_text_md52text and md5 not in compare_result_md52result:
                to_predict_pair_text_md52text[md5] = (text1, text2)
        print('[NEW] to predict pair text:', len(to_predict_pair_text_md52text))

        id_2_text_pair_md5 = {}
        cnt = 0
        all_input = list()
        all_label = list()
        for md5 in to_predict_pair_text_md52text:
            sent_1_text, sent_2_text = to_predict_pair_text_md52text[md5]

            id_2_text_pair_md5[cnt] = md5
            cnt += 1
            id_2_text_pair_md5[cnt] = md5
            cnt += 1

            emb0, emb1 = dic_text_2_embedding[sent_1_text], dic_text_2_embedding[sent_2_text]
            cated_emb = np.concatenate([emb0,emb1], axis=-1)
            all_input.append(cated_emb)
            all_label.append(0)
            cated_emb = np.concatenate([emb1, emb0], axis=-1)
            all_input.append(cated_emb)
            all_label.append(0)

        print('Done: load to input to be predicted')

        if len(all_input) > 0:
            all_input = np.asarray(all_input)
            all_label = np.asarray(all_label)

            input_tensor = torch.from_numpy(all_input)
            label_tensor = torch.from_numpy(all_label)
            train_dataset = TensorDataset(input_tensor, label_tensor)
            train_dataloader = DataLoader(train_dataset, shuffle = False, batch_size = 4096)

            classifier_model = torch.nn.Sequential(OrderedDict(
                [('embedding', train_loss.embedding),
                ('classifier', train_loss.classifier),
                ]
            )).to(device)

            all_probs = list()
            all_preds = list()
            for inputs, labels in train_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = classifier_model(inputs)

                probs = torch.nn.functional.softmax(logits, -1)
                predict = torch.argmax(logits, dim = -1)

                all_probs.append(probs.detach().cpu().numpy())
                all_preds.append(predict.detach().cpu().numpy())

            all_probs = np.concatenate(all_probs,0)
            all_preds = np.concatenate(all_preds,0)

            print('Done: finish predict')

            for i in range(0, len(all_preds), 2):
                md5 = id_2_text_pair_md5[i]
                compare_result_md52result[md5] = False
                if check_predict_and_probability(all_preds, all_probs, i) == True:
                    compare_result_md52result[md5] = True

        with open('../data/EDG_match_result.pkl', 'wb') as f:
            pickle.dump(compare_result_md52result, f)

        match_cnt = 0
        for md5 in compare_result_md52result:
            if compare_result_md52result[md5] == True:
                match_cnt += 1
        print('match cnt total:', match_cnt)

        return compare_result_md52result

    to_predict_pair = get_predict_pair()
    compare_result_md52result = get_compare_result_md52result(True)

    for node_ID, text2 in to_predict_pair:
        text1 = EDG_node[node_ID].content_text
        md5 = ccy.get_md5(text1 + text2)
        if compare_result_md52result[md5] == True:
            EDG_node[node_ID].end_Node = True

    with open('../data/EDG_graph_node.pkl', 'wb') as f:
        pickle.dump(EDG_node, f)

    return


def get_Node_ID(text):

    preparation()

    get_embeddings([text])

    def get_predict_pair(text):

        to_predict_pair = []
        for node_ID in EDG_node:
            if EDG_node[node_ID].cat == 'text':
                to_predict_pair.append((node_ID, text))

        return to_predict_pair

    def get_compare_result_md52result(from_pickle = False):

        def check_predict_and_probability(all_preds, all_probs, i):

            threshold_1 = 0.69
            threshold_23 = 0.9

            if all_preds[i] == 1 and all_preds[i+1] == 1:
                if all_probs[i][1] >= threshold_1 and all_probs[i+1][1] >= threshold_1:
                    return True

            if all_preds[i] == 2 and all_preds[i+1] == 3:
                if all_probs[i][2] >= threshold_23 and all_probs[i+1][3] >= threshold_23:
                    return True

            if all_preds[i] == 3 and all_preds[i+1] == 2:
                if all_probs[i][3] >= threshold_23 and all_probs[i+1][2] >= threshold_23:
                    return True

            return False

        compare_result_md52result = {}
        if from_pickle == True:
            with open('../data/EDG_match_result.pkl', 'rb') as f:
                compare_result_md52result = pickle.load(f)
                print('previous record md5 cnt:', len(compare_result_md52result))

        to_predict_pair_text_md52text = {}
        for node_ID, text2 in to_predict_pair:
            text1 = EDG_node[node_ID].content_text
            md5 = ccy.get_md5(text1+text2)
            if md5 not in to_predict_pair_text_md52text and md5 not in compare_result_md52result:
                to_predict_pair_text_md52text[md5] = (text1, text2)
        print('[NEW] to predict pair text:', len(to_predict_pair_text_md52text))

        id_2_text_pair_md5 = {}
        cnt = 0
        all_input = list()
        all_label = list()
        for md5 in to_predict_pair_text_md52text:
            sent_1_text, sent_2_text = to_predict_pair_text_md52text[md5]

            id_2_text_pair_md5[cnt] = md5
            cnt += 1
            id_2_text_pair_md5[cnt] = md5
            cnt += 1

            emb0, emb1 = dic_text_2_embedding[sent_1_text], dic_text_2_embedding[sent_2_text]
            cated_emb = np.concatenate([emb0,emb1], axis=-1)
            all_input.append(cated_emb)
            all_label.append(0)
            cated_emb = np.concatenate([emb1, emb0], axis=-1)
            all_input.append(cated_emb)
            all_label.append(0)

        print('Done: load to input to be predicted')

        if len(all_input) > 0:
            all_input = np.asarray(all_input)
            all_label = np.asarray(all_label)

            input_tensor = torch.from_numpy(all_input)
            label_tensor = torch.from_numpy(all_label)
            train_dataset = TensorDataset(input_tensor, label_tensor)
            train_dataloader = DataLoader(train_dataset, shuffle = False, batch_size = 4096)

            classifier_model = torch.nn.Sequential(OrderedDict(
                [('embedding', train_loss.embedding),
                ('classifier', train_loss.classifier),
                ]
            )).to(device)

            all_probs = list()
            all_preds = list()
            for inputs, labels in train_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = classifier_model(inputs)

                probs = torch.nn.functional.softmax(logits, -1)
                predict = torch.argmax(logits, dim = -1)

                all_probs.append(probs.detach().cpu().numpy())
                all_preds.append(predict.detach().cpu().numpy())

            all_probs = np.concatenate(all_probs,0)
            all_preds = np.concatenate(all_preds,0)

            print('Done: finish predict')

            for i in range(0, len(all_preds), 2):
                md5 = id_2_text_pair_md5[i]
                compare_result_md52result[md5] = False
                if check_predict_and_probability(all_preds, all_probs, i) == True:
                    compare_result_md52result[md5] = True

        with open('../data/EDG_match_result.pkl', 'wb') as f:
            pickle.dump(compare_result_md52result, f)

        match_cnt = 0
        for md5 in compare_result_md52result:
            if compare_result_md52result[md5] == True:
                match_cnt += 1
        print('match cnt total:', match_cnt)

        return compare_result_md52result

    to_predict_pair = get_predict_pair(text)
    compare_result_md52result = get_compare_result_md52result(True)

    ret_node_ID_list = []

    for node_ID, text2 in to_predict_pair:
        text1 = EDG_node[node_ID].content_text
        md5 = ccy.get_md5(text1 + text2)
        if compare_result_md52result[md5] == True:
            print('Node_ID:', node_ID)
            ret_node_ID_list.append(node_ID)

    return ret_node_ID_list


def debug(flag1 = False, flag2 = False, flag3 = False, flag4 = False):

    with open('../data/EDG_graph_node.pkl', 'rb') as f:
        EDG_node = pickle.load(f)
    with open('../data/EDG_graph_edge.pkl', 'rb') as f:
        EDG_edge = pickle.load(f)


    if flag4 == True:

        sent_1_text = 'sending a SECURITY MODE COMMAND message to the UE'
        sent_2_text = 'send a SECURITY MODE COMMAND message to the UE'
        print(sent_1_text)
        print(sent_2_text)

        new_text_list = list()
        new_text_list.append(sent_1_text)
        new_text_list.append(sent_2_text)

        embeddings = model.encode(new_text_list, batch_size = 1024, show_progress_bar = True)
        emb_dict = {sent: emb for sent, emb in zip(new_text_list, embeddings)}

        dic_text_2_embedding = {}
        for text in new_text_list:
            dic_text_2_embedding[text] = emb_dict[text]

        #with open('../data/EDG_embeddings.pkl', 'rb') as f:
        #    dic_text_2_embedding = pickle.load(f)

        all_input = list()
        all_label = list()

        emb0, emb1 = dic_text_2_embedding[sent_1_text], dic_text_2_embedding[sent_2_text]
        cated_emb = np.concatenate([emb0,emb1], axis=-1)
        all_input.append(cated_emb)
        all_label.append(0)
        cated_emb = np.concatenate([emb1, emb0], axis=-1)
        all_input.append(cated_emb)
        all_label.append(0)

        all_input = np.asarray(all_input)
        all_label = np.asarray(all_label)

        input_tensor = torch.from_numpy(all_input)
        label_tensor = torch.from_numpy(all_label)
        train_dataset = TensorDataset(input_tensor, label_tensor)
        train_dataloader = DataLoader(train_dataset, shuffle = False, batch_size = 4096)

        classifier_model = torch.nn.Sequential(OrderedDict(
            [('embedding', train_loss.embedding),
            ('classifier', train_loss.classifier),
            ]
        )).to(device)

        all_probs = list()
        all_preds = list()
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = classifier_model(inputs)

            probs = torch.nn.functional.softmax(logits, -1)
            predict = torch.argmax(logits, dim = -1)

            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(predict.detach().cpu().numpy())

        all_probs = np.concatenate(all_probs,0)
        all_preds = np.concatenate(all_preds,0)

        def check_predict_and_probability(all_preds, all_probs, i):

            threshold_1 = 0.58
            threshold_23 = 0.9

            print(all_probs[i][1], all_probs[i+1][1])
            print(all_probs[i][2], all_probs[i+1][3])
            print(all_probs[i][3], all_probs[i+1][2])

            if all_preds[i] == 1 and all_preds[i+1] == 1:
                if all_probs[i][1] >= threshold_1 and all_probs[i+1][1] >= threshold_1:
                    return True

            if all_preds[i] == 2 and all_preds[i+1] == 3:
                if all_probs[i][2] >= threshold_23 and all_probs[i+1][3] >= threshold_23:
                    return True

            if all_preds[i] == 3 and all_preds[i+1] == 2:
                if all_probs[i][3] >= threshold_23 and all_probs[i+1][2] >= threshold_23:
                    return True

            return False

        if check_predict_and_probability(all_preds, all_probs, 0) == True:
            print(True)
        else:
            print(False)



def rebuild_graph_connection():
    global EDG_node
    global EDG_edge

    with open('../data/EDG_graph_node.pkl', 'rb') as f:
        EDG_node = pickle.load(f)
    with open('../data/EDG_graph_edge.pkl', 'rb') as f:
        EDG_edge = pickle.load(f)

    '''
    print(EDG_edge[205].__dict__.items())
    print(EDG_edge[7994].__dict__.items())
    print(EDG_edge[204].__dict__.items())
    print(EDG_edge[5655].__dict__.items())
    print(EDG_edge[62].__dict__.items())
    print(EDG_edge[26306].__dict__.items())
    print(EDG_edge[1317].__dict__.items())
    print(EDG_edge[1318].__dict__.items())
    print(EDG_edge[1320].__dict__.items())

    print(EDG_node[2084].__dict__.items())
    print(EDG_node[2087].__dict__.items())
    '''

    # self.ID = ID
    # self.weight = weight
    # self.in_Node_ID = in_Node_ID
    # self.out_Node_ID = out_Node_ID
    gE, fE = dict(), dict()
    for e in EDG_edge.values():
        u, v = e.in_Node_ID, e.out_Node_ID

        if not u in gE:
            gE[u] = list()
        gE[u].append((e.ID, v))
        if not v in fE:
            fE[v] = list()
        fE[v].append((e.ID, u))

    return gE, fE


def find_invocation_route(node, fE, flag):
    q = queue.Queue()
    q.put(node)

    visit = dict()

    st_nodes = list()
    act_edges = list()
    while not q.empty():
        v = q.get()
        visit[v] = 2

        if EDG_node[v].start_Node == True:
            st_nodes.append(v)
            continue

        if not v in fE:
            continue

        m = 0
        for e, u in fE[v]:
            if not u in flag:
                continue
            if EDG_node[u].f < 1:
                continue
            if EDG_node[u].dist_st >= EDG_node[v].dist_st:
                continue
            m += EDG_edge[e].weight
            act_edges.append(e)
            if not u in visit:
                q.put(u)
                visit[u] = 1
            if m >= EDG_node[v].weight:
                break

    return st_nodes, act_edges


def find_observation_route(node, gE, flag):
    q = queue.Queue()
    q.put(node)

    visit = dict()

    obs_nodes = list()
    act_edges = list()
    while not q.empty():
        u = q.get()
        visit[u] = 2

        if EDG_node[u].end_Node == True:
            obs_nodes.append(u)
            continue

        if not u in gE:
            continue

        # m = 0
        for e, v in gE[u]:
            if not v in flag:
                continue
            if EDG_node[v].f < 1:
                continue
            # if EDG_node[v].dist_st <= EDG_node[u].dist_st:
            #    continue
            # m += EDG_edge[e].weight
            act_edges.append(e)
            if not v in visit:
                q.put(v)
                visit[v] = 1

    return obs_nodes, act_edges





def forward_search(st_nodes, gE, thr, visit=None, return_flag=False):
    q = queue.Queue()
    for u in st_nodes:
        EDG_node[u].msg = EDG_node[u].weight
        EDG_node[u].dist_st = 0
        q.put(u)

    obs_list = list()
    flag = dict()
    while not q.empty():
        u = q.get()
        flag[u] = 2

        if EDG_node[u].end_Node == True:
            obs_list.append(u)

        if EDG_node[u].msg >= EDG_node[u].weight:
            EDG_node[u].f = 1
        else:
            EDG_node[u].f = 0
            continue

        if not u in gE:
            continue

        for e, v in gE[u]:
            if visit is not None and not v in visit:
                continue
            if EDG_edge[e].weight < thr:
                continue
            if not hasattr(EDG_node[v], "msg"):
                EDG_node[v].msg = 0
            EDG_node[v].msg += EDG_edge[e].weight

            if EDG_node[v].msg >= EDG_node[v].weight:
                if not hasattr(EDG_node[v], "dist_st"):
                    EDG_node[v].dist_st = EDG_node[u].dist_st+1
                if not v in flag:
                    flag[v] = 1
                    q.put(v)

    if return_flag:
        return obs_list, flag
    return obs_list



def observable(node, thr=0.5):
    gE, fE = rebuild_graph_connection()

    # node_weight = EDG_node[node].weight
    # EDG_node[node].weight = len(EDG_edge)+1

    st_nodes = [node]
    for no in EDG_node.values():
        if no.start_Node == True:
            st_nodes.append(no.ID)
    st_nodes = [node]

    _, flag = forward_search(st_nodes, gE, thr, visit=None, return_flag=True)


    '''
    def_obs = dict()
    for no in EDG_node.values():
        if hasattr(no,'f') and no.f == 1 and no.end_Node == True:
            def_obs[no.ID] = 1

    EDG_node[node].weight = node_weight
    _, flag = forward_search([node], gE, thr, visit=None, return_flag=True)
    '''

    _obs_nodes, _act_edges = find_observation_route(node, gE, flag)

    '''
    _obs_nodes = list()
    for o in obs_nodes:
        if o in def_obs:
            continue
        _obs_nodes.append(o)
    '''
    _obs_nodes.sort()
    print(_obs_nodes)
    # print(_act_edges)

    # for e in _act_edges:
    #     print(EDG_edge[e].in_Node_ID, '->', EDG_edge[e].out_Node_ID)


def invocable(ids, thr=1):
    gE, fE = rebuild_graph_connection()

    # for nid in EDG_node:
        # EDG_node[nid].start_Node =False

    q = queue.Queue()
    for i in ids:
        q.put(i)

    st_nodes = list()
    visit = dict()
    while not q.empty():
        v = q.get()

        if EDG_node[v].start_Node == True and not v in visit:
            st_nodes.append(v)

        visit[v] = True
        if EDG_node[v].start_Node == True:
            continue

        if not v in fE:
            continue

        for e, u in fE[v]:
            if EDG_edge[e].weight < thr:
                continue
            if not u in visit:
                q.put(u)


    # print(st_nodes)
    # exit(0)

    _, flag = forward_search(st_nodes, gE, thr, visit=visit, return_flag=True)

    print(EDG_node[1168].__dict__.items())
    print(EDG_node[1167].__dict__.items())
    print(EDG_node[1166].__dict__.items())
    print(fE[1166])
    print(EDG_node[1170].__dict__.items())
    for i in ids:
        if hasattr(EDG_node[i], "f") and EDG_node[i].f > 0:
            print(i)

    _st_nodes, _act_edges = find_invocation_route(ids[0], fE, flag)
    print(_st_nodes)
    print(_act_edges)

    for e in _act_edges:
        print(EDG_edge[e].in_Node_ID, '->', EDG_edge[e].out_Node_ID)








if __name__ == '__main__':

    #preparation()

    #build_start_Node_set()
    #build_end_Node_set()

    # SR_condition_text = 'the encryption of NAS messages has been started between the MME and the UE'
    # SR_result_text = 'the UE shall start timer T3402'

    # SR_condition_text = 'UE is required to delete an eKSI'

    # node_ID_list = get_Node_ID(SR_condition_text)
    # node_ID_list = [72] 
    # node_ID_list = [85]
    # node_ID_list = [79]
    # node_ID_list = [2373]
    # node_ID_list = [333]
    # node_ID_list = [335]
    # node_ID_list = [6230]
    # node_ID_list = [6280]
    # invocable(node_ID_list)

    # node_ID = 2204
    # node_ID = 2217
    node_ID = 6278
    observable(node_ID)

    #debug(False, False, False, True)
