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

SEED=910911

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(SEED)


def loading_training_set_from_file(filepath_lst, num, max_sent_words = 128):

    train_examples = []


    fileContent = []
    for filepath in filepath_lst:

        file = open(filepath)
        fileContent_tmp = file.readlines()
        fileContent += fileContent_tmp
        file.close()

    index_list = list()
    for i in range(0, len(fileContent), 3):
        sentence_1 = fileContent[i+1].strip()
        sentence_2 = fileContent[i+2].strip()
        len1 = len(sentence_1.split(' '))
        len2 = len(sentence_2.split(' '))
        if max(len1, len2) <= max_sent_words:
            index_list.append(i)

    index_list = np.asarray(index_list)
    rep_n = math.ceil(num/len(index_list))
    _list = list()
    for _ in range(rep_n):
        _list.append(np.random.permutation(index_list))
    index_list = np.concatenate(_list)
    index_list = index_list[:num]

    for i in index_list:
        lb = int(fileContent[i].strip())
        sentence_1 = fileContent[i+1].strip()
        sentence_2 = fileContent[i+2].strip()
        if lb == 0 or lb == 1:
            train_examples.append(InputExample(texts = [sentence_1, sentence_2], label = lb))
            #train_examples.append(InputExample(texts = [sentence_2, sentence_1], label = lb))
        elif lb == 2:
            train_examples.append(InputExample(texts = [sentence_1, sentence_2], label = 2))
            #train_examples.append(InputExample(texts = [sentence_2, sentence_1], label = 3))


    return train_examples


def loading_training_set():

    num = 200000
    data_list = [
        {'filepath': ['../data/training_set_for_label_0'], 'num': num},
        {'filepath': ['../data/training_set_for_label_1'], 'num': num},
        {'filepath': ['../data/training_set_for_label_2'], 'num': num * 2},
    ]

    train_examples = []
    for data in data_list:
        filepath, num = data['filepath'], data['num']
        train_examples += loading_training_set_from_file(filepath, num)

    return train_examples


def loading_evaluation_set(flag, filepath='../data/evaluation_set.txt'):

    file = open(filepath)
    # file = open('../data/evaluation_set.txt')
    #file = open('../data/evaluation_set.recall')
    fileContent = file.readlines()
    file.close()

    if flag == 'for_train_all_model':

        sentence_1_list = []
        sentence_2_list = []
        label_list = []

        for i in range(0, len(fileContent), 3):
            label = int(fileContent[i].strip())
            sentence_1 = fileContent[i+1].strip()
            sentence_2 = fileContent[i+2].strip()
            sentence_1_list.append(sentence_1)
            sentence_2_list.append(sentence_2)
            label_list.append(label)


        return (sentence_1_list, sentence_2_list, label_list)

    elif flag == 'for_evaluation':

        train_examples = []

        for i in range(0, len(fileContent), 3):
            lb = int(fileContent[i].strip())
            sentence_1 = fileContent[i+1].strip()
            sentence_2 = fileContent[i+2].strip()
            train_examples.append(InputExample(texts = [sentence_1, sentence_2], label = lb))

        return train_examples


def train_all_model(epochs=20, load_embeddings=False, no_last_evaluation=False, only_evaluation=False, no_tqdm=False, only_evaluate_recall=False, **kwargs):

    if only_evaluate_recall:
        evaluation('[only recall]', for_recall=True)
        return
    if only_evaluation:
        evaluation('[only classifier]')
        return

    # '''
    batch_size = 4096
    device = 'cuda'

    model = SentenceTransformer('all-mpnet-base-v2')
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    train_loss = CosineMimicLoss(model, feature_dim = model.get_sentence_embedding_dimension())
    train_loss = train_loss.to(device)

    sentence_1_list, sentence_2_list, label_list = loading_evaluation_set('for_train_all_model')
    myevaluator = myEvaluator(sentence_1_list, sentence_2_list, label_list, loss_model = train_loss, batch_size = batch_size)
    # callback_fn = get_callback_save_fn(train_loss, '../data/model_part1.pt', demo_fn=evaluation)
    callback_fn = get_callback_save_fn(train_loss, '../data/model_part1.pt', demo_fn=None, seed=kwargs['seed'])

    train_examples = loading_training_set()

    if load_embeddings:
        with open('embeddings.pkl','rb') as f:
            emb_dict = pickle.load(f)
        print('-'*20, 'load emb_dict from embeddings.pkl', '-'*20)
    else:
        all_sents = list()
        for train_example in train_examples:
            all_sents.append(train_example.texts[0])
            all_sents.append(train_example.texts[1])
        all_sents = list(set(all_sents))
        all_sents.sort()
        print('-'*20, 'all %d sentences are loaded'%(len(all_sents)), '-'*20)

        embeddings = model.encode(all_sents, batch_size=1024, show_progress_bar=not no_tqdm)

        print('-'*20, 'all {} sentences are converted to tensor with shape {}'.format(len(all_sents), embeddings.shape), '-'*20)

        emb_dict = {sent: emb for sent, emb in zip(all_sents, embeddings)}

        with open('embeddings.pkl','wb') as f:
            pickle.dump(emb_dict, f)
        print('-'*20, 'save emb_dict to embeddings.pkl', '-'*20)

    all_input = list()
    all_label = list()
    for train_example in train_examples:
        label = train_example.label
        sent0, sent1 = train_example.texts
        emb0, emb1 = emb_dict[sent0], emb_dict[sent1]
        cated_emb = np.concatenate([emb0,emb1], axis=-1)
        all_input.append(cated_emb)
        all_label.append(label)

        rever_emb = np.concatenate([emb1,emb0], axis=-1)
        if label==2:
            rever_lab = 3
        elif label==3:
            rever_lab = 2
        else:
            rever_lab = label
        all_input.append(rever_emb)
        all_label.append(rever_lab)
    all_input = np.asarray(all_input)
    all_label = np.asarray(all_label)
    print('-'*20, 'trainin set were built: {}, {}'.format(all_input.shape, all_label.shape), '-'*20)

    input_tensor = torch.from_numpy(all_input)
    label_tensor = torch.from_numpy(all_label)
    train_dataset = TensorDataset(input_tensor, label_tensor)
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=batch_size)

    classifier_model = torch.nn.Sequential(OrderedDict(
        [('embedding', train_loss.embedding),
         ('classifier', train_loss.classifier),
         ]
    )).to(device)
    optim = torch.optim.Adam(classifier_model.parameters(), lr=0.01, betas=(0.9,0.98))
    evaluation_steps = 0

    pbar = range(epochs) if no_tqdm else tqdm(range(epochs))
    for epoch in pbar:
        step = 0
        classifier_model.train()
        #for inputs, labels in tqdm(train_dataloader):
        for inputs, labels in train_dataloader:
            step += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optim.zero_grad()

            logits = classifier_model(inputs)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optim.step()

            #if evaluation_steps > 0 and (step % evaluation_steps) == 0:
            #    sc = myevaluator(model, epoch=epoch, steps=step-1)
            #    callback_fn(sc, epoch, step-1)
            #    classifier_model.train()

        sc = myevaluator(model, epoch=epoch, steps=step-1)
        if epoch == 0 and sc < 50: break
        if epoch == 3 and sc < 70: break
        callback_fn(sc, epoch, step-1)


    # '''

    if not no_last_evaluation:
        evaluation('[only classifier]')


def td_evaluation(log=None):
    if log is None:
        log = '[test]'
    file = open('../data/evaluation.log', 'a')

    file.write('%s\n\n'%log)

    device='cuda'
    model = SentenceTransformer('../data/model_part1.pt', device=device)
    train_loss = torch.load('../data/model_part2.pt', map_location = device)
    train_loss.model = model
    train_loss = train_loss.to(device)

    sentence_1_list, sentence_2_list, label_list = loading_evaluation_set('for_train_all_model')
    myevaluator = myEvaluator(sentence_1_list, sentence_2_list, label_list, loss_model = train_loss, batch_size = 256)

    ce, acc, probs, preds, labels = myevaluator.compute_ce_score(model, return_probs=True)

    for i in range(len(labels)):
        z = '{id}:    {p0:.2f} {p1:.2f} {p2:.2f} {p3:.2f}    {pred}:{label} (pred:label)'.format(
            id=i, p0=probs[i][0], p1=probs[i][1], p2=probs[i][2], p3=probs[i][3], pred=preds[i], label=labels[i]
        )
        print(z)

        file.write(str(z)+'\n\n')

    cnt_all = len(labels)
    cnt_correct = np.sum(preds==labels)
    print('{:d} correct, {:d} wrong'.format(cnt_correct, cnt_all - cnt_correct))

    file.write('%d correct, %d wrong\n\n\n'%(cnt_correct, cnt_all - cnt_correct))
    file.close()


def evaluation(log=None, for_recall=False):

    if log is None:
        log = '[test]'
    file = open('../data/evaluation.log', 'a')

    file.write('%s\n\n'%log)

    device='cuda'
    model = SentenceTransformer('../data/model_part1.pt', device=device)
    train_loss = torch.load('../data/model_part2.pt', map_location = device)
    train_loss.model = model
    train_loss = train_loss.to(device)

    if for_recall:
        train_examples = loading_evaluation_set('for_evaluation', filepath='../data/evaluation_set.recall')
    else:
        train_examples = loading_evaluation_set('for_evaluation')

    train_dataloader = DataLoader(train_examples, shuffle = False, batch_size = 128)
    train_dataloader.collate_fn = model.smart_batching_collate

    train_loss.set_predict()
    train_loss.eval()

    cnt_all = 0
    cnt_correct = 0

    labels = list()
    preds = list()
    probs = list()
    for data in train_dataloader:
        sentence_batch, label = data

        label = label.to(device)
        sentence_batch = list(map(lambda batch: batch_to_device(batch, device), sentence_batch))

        output = train_loss(sentence_batch, label)

        predict = torch.argmax(output, dim = 1)
        rst = torch.eq(predict, label)
        cnt_all += len(rst)
        cnt_correct += torch.sum(rst).detach().cpu().numpy()

        labels.append(label.detach().cpu().numpy())
        preds.append(predict.detach().cpu().numpy())
        probs.append(output.detach().cpu().numpy())

    labels = np.concatenate(labels, 0)
    preds = np.concatenate(preds, 0)
    probs = np.concatenate(probs, 0)
    for i in range(len(labels)):
        z = '{id}:    {p0:.2f} {p1:.2f} {p2:.2f} {p3:.2f}    {pred}:{label} (pred:label)'.format(
            id=i, p0=probs[i][0], p1=probs[i][1], p2=probs[i][2], p3=probs[i][3], pred=preds[i], label=labels[i]
        )
        print(z)

        file.write(str(z)+'\n')


    print('{:d} correct, {:d} wrong'.format(cnt_correct, cnt_all - cnt_correct))

    file.write('%d correct, %d wrong\n\n\n'%(cnt_correct, cnt_all - cnt_correct))
    file.close()



def get_argparse():
    parser = argparse.ArgumentParser('train_all_model')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--random_seed', action='store_true', default=False)
    parser.add_argument('--load_embeddings', action='store_true', default=False)
    parser.add_argument('--no_last_evaluation', action='store_true', default=False)
    parser.add_argument('--only_evaluation', action='store_true', default=False)
    parser.add_argument('--only_evaluate_recall', action='store_true', default=False)
    parser.add_argument('--no_tqdm', action='store_true', default=False)

    return parser

if __name__ == '__main__':

    parser = get_argparse()
    configs = parser.parse_args()
    configs = vars(configs)

    if configs['random_seed']:
        random.seed()
        configs['seed'] = random.randint(100001, 999999)

    if configs['seed']:
        set_seed(configs['seed'])

    train_all_model(**configs)

