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

    if True:
        if True:
            if True:

                with open('../exp/evaluation_set.txt', 'r') as f:
                    fileContent = f.readlines()

                s1 = []
                s2 = []

                train_examples = []
                for i in range(0, len(fileContent), 3):
                    s1 = fileContent[i+1].strip()
                    s2 = fileContent[i+2].strip()

                    train_examples.append(InputExample(texts = [s1, s2], label = 0))




                train_dataloader = DataLoader(train_examples, shuffle = False, batch_size = 16)
                train_dataloader.collate_fn = model.smart_batching_collate

                for data in train_dataloader:
                    sentence_batch, label = data
                    output = train_loss(sentence_batch, label).detach()
                    predict = torch.argmax(output, dim = 1)
                    print(output)
                    print(predict)

if __name__ == '__main__':

    sent_text = 'a partial native EPS security context is taken into use through a security mode control procedure'

    get_classification(sent_text)
