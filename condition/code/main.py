import pickle

from TD.CosineMimicLoss import CosineMimicLoss
from TD.CosineMimicLoss import myEvaluator
from TD.CosineMimicLoss import get_callback_save_fn

import torch
from sentence_transformers import SentenceTransformer, InputExample, models, util
from torch.utils.data import DataLoader
from typing import Iterable, Dict
from torch import Tensor
from sentence_transformers.util import batch_to_device
import torch.nn.functional as F
import numpy as np
from datetime import datetime

import NLP.data


dic_sent = {}
with open('../data/condition_result_for_sentences.pickle', 'rb') as f:
    dic_sent = pickle.load(f)


model = SentenceTransformer('../data/model_part1.pt')
train_loss = torch.load('../data/model_part2.pt', map_location = model.device)
train_loss.model = model
train_loss.set_predict()
train_loss.eval()


cosine_model = SentenceTransformer('all-mpnet-base-v2')


def get_classification(sent_text):

    embedding_sent_text = cosine_model.encode([sent_text], convert_to_tensor = True)

    classification = {1:[], 2:[], 3:[]}

    cnt = 0
    cnt_cosine = 0
    cnt_our_model = 0
    for md5 in dic_sent:
        for x in dic_sent[md5]['mini_tree_set']:
            if x.result != None:

                #'''
                embedding_result_text = cosine_model.encode([x.result], convert_to_tensor = True)
                cosine_scores = util.cos_sim(embedding_sent_text, embedding_result_text)
                #if cosine_scores[0][0] <= 0.5:
                #    continue
                #'''

                if cosine_scores[0][0] >= 0.5:
                    cnt_cosine += 1

                train_examples = [InputExample(texts = [sent_text, x.result], label = 0)]
                train_dataloader = DataLoader(train_examples, shuffle = False, batch_size = 16)
                train_dataloader.collate_fn = model.smart_batching_collate

                cls = None
                for data in train_dataloader:
                    sentence_batch, label = data
                    output = train_loss(sentence_batch, label).detach()
                    predict = torch.argmax(output, dim = 1)
                    cls = predict.item()
                    break

                assert cls != None


                #print(cnt, '/', len(dic_sent))
                print(x.result)
                print(sent_text)
                print(cls)
                print(output)
                print(cosine_scores[0][0])
                print(' ')

                if cls != 0:
                    cnt_our_model += 1
                    classification[cls].append(x.result)
        cnt = cnt + 1

    print(cnt_cosine, cnt_our_model)


if __name__ == '__main__':

    sent_text = 'a partial native EPS security context is taken into use through a security mode control procedure'

    get_classification(sent_text)
