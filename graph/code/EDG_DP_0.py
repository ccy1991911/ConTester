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

    for node_ID, text2 in to_predict_pair:
        text1 = EDG_node[node_ID].content_text
        md5 = ccy.get_md5(text1 + text2)
        if compare_result_md52result[md5] == True:
            print('Node_ID:', node_ID)

    return node_ID


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



if __name__ == '__main__':

    preparation()

    build_start_Node_set()
    build_end_Node_set()

    #SR_condition_text = 'the encryption of NAS messages has been started between the MME and the UE'
    #SR_result_text = 'the UE shall start timer T3402'

    #get_Node_ID(SR_condition_text)

    #debug(False, False, False, True)
