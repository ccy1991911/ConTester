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

class Node():
    def __init__(self, ID, weight, cat, condition_or_result, content_text, sent_text):

        self.ID = ID
        self.weight = weight
        self.cat = cat
        self.condition_or_result = condition_or_result
        self.content_text = content_text
        self.sent_text = sent_text

class Edge():
    def __init__(self, ID, weight, in_Node_ID, out_Node_ID):

        self.ID = ID
        self.weight = weight
        self.in_Node_ID = in_Node_ID
        self.out_Node_ID = out_Node_ID

def create_Node(weight, cat, condition_or_result = None, context_text = None, sent_text = None):
    ID = len(EDG_node) + 1
    EDG_node[ID] = Node(ID, weight, cat, condition_or_result, context_text, sent_text)
    return ID

def create_Edge(weight, in_Node_ID, out_Node_ID):
    ID = len(EDG_edge) + 1
    EDG_edge[ID] = Edge(ID, weight, in_Node_ID, out_Node_ID)
    return ID

# EDG data
EDG_node = {}
EDG_edge = {}
def build_EDG_for_only_causal_relation():

    # from condition_result_for_sentences.pickle
    dic_md5 = {}
    with open('../data/condition_result_for_sentences.pickle', 'rb') as f:
        dic_md5 = pickle.load(f)

    for md5 in dic_md5:
        sent_text = dic_md5[md5]['sent_text']
        mini_tree_set = dic_md5[md5]['mini_tree_set']
        for mt in mini_tree_set:
            conditions = mt.conditions
            result = mt.result

            if result != None:
                len_conditions = len(conditions)
                if len_conditions == 1:
                    condition_text, introducer_word = conditions[0]
                    condition_Node_ID = create_Node(0, 'text', 'condition', condition_text, sent_text)
                    result_Node_ID = create_Node(1, 'text', 'result', result, sent_text)
                    create_Edge(1, condition_Node_ID, result_Node_ID)

                else:

                    and_Node_ID = create_Node(len(conditions), 'AND')
                    result_Node_ID = create_Node(1, 'text', 'result', result, sent_text)
                    create_Edge(1, and_Node_ID, result_Node_ID)

                    for condition_text, introducer_word in conditions:
                        condition_Node_ID = create_Node(0, 'text', 'condition', condition_text, sent_text)
                        create_Edge(1, condition_Node_ID, and_Node_ID)

                        if introducer_word == 'by':
                            condition_Node_ID = create_Node(0, 'text', 'condition', result, sent_text)
                            result_Node_ID = create_Node(1, 'text', 'result', condition_text, sent_text)
                            create_Edge(1, condition_Node_ID, result_Node_ID)

    print(len(EDG_node))
    print(len(EDG_edge))

    with open('../data/EDG_graph_node.pkl', 'wb') as f:
            pickle.dump(EDG_node, f)
    with open('../data/EDG_graph_edge.pkl', 'wb') as f:
            pickle.dump(EDG_edge, f)

    print('Done: build EDG for only causal relation')

device='cuda'
model = SentenceTransformer('../data/model_part1.pt', device=device)
train_loss = torch.load('../data/model_part2.pt', map_location = device)
train_loss.model = model
train_loss = train_loss.to(device)
train_loss.set_predict()
train_loss.eval()


def build_EDG_with_model(from_pickle):

    def get_embeddings(from_pickle = False):

        dic_text_2_embedding = {}

        if from_pickle == True:
            with open('../data/EDG_embeddings.pkl', 'rb') as f:
                dic_text_2_embedding = pickle.load(f)

        new_text_list = list()

        for node_ID in EDG_node:
            if EDG_node[node_ID].cat == 'text':
                content_text = EDG_node[node_ID].content_text
                if content_text not in dic_text_2_embedding:
                    new_text_list.append(content_text)

        print('new text: ', len(new_text_list))

        embeddings = model.encode(new_text_list, batch_size = 1024, show_progress_bar = True)
        emb_dict = {sent: emb for sent, emb in zip(new_text_list, embeddings)}

        for text in new_text_list:
            dic_text_2_embedding[text] = emb_dict[text]

        with open('../data/EDG_embeddings.pkl', 'wb') as f:
            pickle.dump(dic_text_2_embedding, f)

        return dic_text_2_embedding

    def get_predict_pair_node_ID():

        to_predict_pair_node_ID = []
        for node_result_ID in EDG_node:
            if EDG_node[node_result_ID].condition_or_result != 'result':
                continue
            for node_condition_ID in EDG_node:
                if EDG_node[node_condition_ID].condition_or_result != 'condition':
                    continue
                if EDG_node[node_result_ID].sent_text == EDG_node[node_condition_ID].sent_text:
                    continue
                to_predict_pair_node_ID.append((node_result_ID, node_condition_ID))
        print('to_predict_pair_node_ID:', len(to_predict_pair_node_ID))

        return to_predict_pair_node_ID

    def get_compare_result_md52result(from_pickle = False):

        def check_predict_and_probability(all_preds, all_probs, i):

            threshold_1 = 0.58
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
        print('compare_result_md52result:', len(compare_result_md52result))

        to_predict_pair_text_md52text = {}
        for node_result_ID, node_condition_ID in to_predict_pair_node_ID:
            text_result = EDG_node[node_result_ID].content_text
            text_condition = EDG_node[node_condition_ID].content_text
            md5 = ccy.get_md5(text_result+text_condition)
            if md5 not in to_predict_pair_text_md52text and md5 not in compare_result_md52result:
                to_predict_pair_text_md52text[md5] = (text_result, text_condition)
        print('to_predict_pair_text:', len(to_predict_pair_text_md52text))

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

    dic_text_2_embedding = get_embeddings(from_pickle)
    to_predict_pair_node_ID = get_predict_pair_node_ID()
    compare_result_md52result = get_compare_result_md52result(from_pickle)

    for node_result_ID, node_condition_ID in to_predict_pair_node_ID:
        text_result = EDG_node[node_result_ID].content_text
        text_condition = EDG_node[node_condition_ID].content_text
        md5 = ccy.get_md5(text_result+text_condition)

        if compare_result_md52result[md5] == True:
            create_Edge(1, node_result_ID, node_condition_ID)

    with open('../data/EDG_graph_node.pkl', 'wb') as f:
            pickle.dump(EDG_node, f)
    with open('../data/EDG_graph_edge.pkl', 'wb') as f:
            pickle.dump(EDG_edge, f)


def debug(flag1 = False, flag2 = False, flag3 = False, flag4 = False):

    with open('../data/EDG_graph_node.pkl', 'rb') as f:
        EDG_node = pickle.load(f)
    with open('../data/EDG_graph_edge.pkl', 'rb') as f:
        EDG_edge = pickle.load(f)

    def output_EDG():

        def find_next_Node_ID(AND_Node_ID):

            for ID in EDG_edge:
                if EDG_edge[ID].in_Node_ID == AND_Node_ID:
                    return EDG_edge[ID].out_Node_ID

            print('error')

        for ID in EDG_edge:
            in_Node_ID = EDG_edge[ID].in_Node_ID
            out_Node_ID = EDG_edge[ID].out_Node_ID

            if EDG_node[in_Node_ID].cat == 'text' and EDG_node[out_Node_ID].cat == 'text':
                print('\n\n')
                print('-----')
                print('edge_ID:', ID)
                print('node_ID: %d --> node_ID: %d'%(in_Node_ID, out_Node_ID))
                print('\nin_text:', EDG_node[in_Node_ID].content_text)
                print('\nin_source:', EDG_node[in_Node_ID].sent_text)
                print('\nout_text:', EDG_node[out_Node_ID].content_text)
                print('\nout_source:', EDG_node[out_Node_ID].sent_text)
            elif EDG_node[in_Node_ID].cat == 'text' and EDG_node[out_Node_ID].cat == 'AND':
                next_Node_ID = find_next_Node_ID(out_Node_ID)
                print('\n\n')
                print('----')
                print('edge_ID:', ID)
                print('node_ID: %d --> AND --> node_ID: %d'%(in_Node_ID, next_Node_ID))
                print('\nin_text:', EDG_node[in_Node_ID].content_text)
                print('\nout_text:', EDG_node[next_Node_ID].content_text)
                print('\nin_source:', EDG_node[in_Node_ID].sent_text)
                print('\nout_source:', EDG_node[next_Node_ID].sent_text)

    def get_edge_text(ID):
        in_Node_ID = EDG_edge[ID].in_Node_ID
        out_Node_ID = EDG_edge[ID].out_Node_ID

        in_text = EDG_node[in_Node_ID].sent_text if EDG_node[in_Node_ID].cat == 'text' else None
        out_text = EDG_node[out_Node_ID].sent_text if EDG_node[out_Node_ID].cat == 'text' else None

        return (in_text, out_text)


    def query1(sent_text):

        for ID in EDG_edge:
            in_text, out_text = get_edge_text(ID)
            if in_text == out_text and in_text == sent_text:
                return ID

        return None

    def query2(in_Node_ID, out_Node_ID):

        for ID in EDG_edge:
            if EDG_edge[ID].in_Node_ID == in_Node_ID and EDG_edge[ID].out_Node_ID == out_Node_ID:
                return ID

        return None

    if flag1 == True:
        output_EDG()

    if flag2 == True:
        s = 'Once the encryption of NAS messages has been started between the MME and the UE, the receiver shall discard the unciphered NAS messages which shall have been ciphered according to the rules described in this specification.'
        s = 'Except for the CONTROL PLANE SERVICE REQUEST message including an ESM message container information element or a NAS message container information element, the UE shall start the ciphering and deciphering of NAS messages when the secure exchange of NAS messages has been established for a NAS signalling connection.'
        s = 'Secure exchange of NAS messages via a NAS signalling connection is usually established by the MME during the attach procedure by initiating a security mode control procedure.'
        #s = 'The MME initiates the NAS security mode control procedure by sending a SECURITY MODE COMMAND message to the UE and starting timer T3460.'
        #s = 'If UE starts timer T3346, the timer T3346 will expire after the period of time as specified by T3346.'
        print('query 1:', query1(s))

    if flag3 == True:
        in_Node_ID = 1315
        out_Node_ID = 2375
        print('query 2:', query2(in_Node_ID, out_Node_ID))

    if flag4 == True:
        in_Node_ID = 1315
        out_Node_ID = 2375

        sent_1_text = EDG_node[in_Node_ID].content_text
        sent_2_text = EDG_node[out_Node_ID].content_text
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

    build_EDG_for_only_causal_relation()
    build_EDG_with_model(False)
    #debug(False, False, False, False)
    print('Done all code')
