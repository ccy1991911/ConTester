from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
from typing import Iterable, Dict, List
from torch import Tensor
import numpy as np
from tqdm.autonotebook import trange
import logging


def compute_kl_loss(p: Tensor, q: Tensor):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2.0
    return loss


class CosineMimicLoss(torch.nn.Module):
    def __init__(self, model: SentenceTransformer, feature_dim: int, parallel: bool = False):
        super(CosineMimicLoss, self).__init__()
        self.model = model
        self._target_device = model._target_device
        self.feature_dim = feature_dim
        self.embedding = self.get_mlp_model(input_dims=feature_dim*2, feature_dims=[512,256])
        self.classifier = self.get_mlp_model(input_dims=256, feature_dims=[256,4])
        self.regression = self.get_mlp_model(input_dims=256, feature_dims=[256,1])
        self.mode = 0
        self.train_classifier_only = False
        self.info_gain = 1e-2

        self.parallel = parallel
        if parallel:
            self.model = nn.DataParallel(self.model)
            self.embedding = nn.DataParallel(self.embedding)
            self.classifier = nn.DataParallel(self.classifier)
            self.regression = nn.DataParallel(self.regression)


    def get_mlp_model(self, input_dims:int, feature_dims: Iterable[int]):
        list_layer = list()
        last_dim = input_dims
        for i, dim in enumerate(feature_dims):
            fn = torch.nn.Linear(in_features=last_dim, out_features=dim)
            list_layer.append(('fn{:d}'.format(i + 1), fn))
            if i == 0:
                dropout = torch.nn.Dropout(p=0.4)
                list_layer.append(('dropout{:d}'.format(i + 1), dropout))
            relu = torch.nn.ReLU()
            list_layer.append(('relu{:d}'.format(i + 1), relu))
            last_dim = dim
        _module = torch.nn.Sequential(OrderedDict(list_layer)).to(self._target_device)
        return _module

    def set_train_cosine(self):
        self.mode = 1

    def set_train_dual(self):
        self.mode = 2

    def set_train_dual_and_regression(self):
        self.mode = 3

    def set_train_dual_rdrop(self):
        self.mode = 4

    def set_predict(self):
        self.mode = 0

    def set_train_classifier_only(self):
        self.train_classifier_only = True
        self.model.eval()

    def reset_train_classifier_only(self):
        self.train_classifier_only = False

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor = None):
        if self.mode == 1:
            outputs = self._forward(sentence_features, return_reps=True)

            outputs, reps0, reps1 = outputs
            cosine_value: Tensor = torch.cosine_similarity(reps0, reps1)
            cosine_value = cosine_value / 2.0 + 0.5

            targets = torch.zeros_like(outputs)
            targets[:, 0] = 1 - cosine_value
            targets[:, 1] = cosine_value

            loss = torch.nn.functional.cross_entropy(outputs, targets.data)
            return loss
        elif self.mode == 2:
            label1 = labels
            output1_0, output2_0 = self._forward(sentence_features, return_dual=True)

            ce_loss1 = F.cross_entropy(output1_0, label1)

            idx2 = labels.eq(2)
            idx3 = labels.eq(3)

            label2 = labels.clone()
            label2[idx2] = 3
            label2[idx3] = 2

            ce_loss2 = F.cross_entropy(output2_0, label2)

            return (ce_loss1 + ce_loss2)*0.5

        elif self.mode == 4:
            label1 = labels
            output1_0, output2_0, output1_1, output2_1 = self._forward(sentence_features, return_dual=True, return_double=True)

            ce_loss1 = (F.cross_entropy(output1_0, label1) + F.cross_entropy(output1_1, label1)) * 0.5
            kl_loss1 = compute_kl_loss(output1_0, output1_1)
            loss1 = ce_loss1 + 0.5 * kl_loss1

            idx2 = labels.eq(2)
            idx3 = labels.eq(3)

            label2 = labels.clone()
            label2[idx2] = 3
            label2[idx3] = 2

            ce_loss2 = (F.cross_entropy(output2_0, label2) + F.cross_entropy(output2_1, label2)) * 0.5
            kl_loss2 = compute_kl_loss(output2_0, output2_1)
            loss2 = ce_loss2 + 0.5 * kl_loss2

            return loss1+loss2

        elif self.mode == 3:
            outputs, s1, output2, s2 = self._forward(sentence_features, return_dual=True, return_regr=True)

            idx0 = labels.eq(0)
            idx1 = labels.eq(1)
            idx2 = labels.eq(2)
            idx3 = labels.eq(3)

            label2 = labels.clone()
            label2[idx2] = 3
            label2[idx3] = 2

            loss1 = F.cross_entropy(outputs, labels)
            loss2 = F.cross_entropy(output2, label2)
            loss_ce = loss1+loss2/2.0

            loss_sc = 0
            gain = self.info_gain

            loss_sc += torch.sum(F.relu(s1[idx0]-gain) *4 + F.relu(s2[idx0]-gain) *4)
            loss_sc += torch.sum(F.relu(1-s1[idx1]) *4 + F.relu(1-s2[idx1]) *4)
            # loss_sc += torch.sum(F.relu(gain-s1[idx2]) *3 + F.relu(1-s2[idx2]) *3 + F.relu(s1[idx2]-(s2[idx2].data-gain)) *2)
            # loss_sc += torch.sum(F.relu(1-s1[idx3]) *3 + F.relu(gain-s2[idx3]) *3 + F.relu(s2[idx3]-(s1[idx3].data-gain)) *2)
            loss_sc /= 8*len(s1)

            return loss_ce + 0e-3 * loss_sc
        elif self.mode == 0:
            outputs = self._forward(sentence_features)
            outputs = F.softmax(outputs, dim=-1)
            return outputs
        else:
            raise NotImplementedError

    def _forward(self, sentence_features: Iterable[Dict[str, Tensor]], return_reps: bool = False, return_dual: bool = False, return_regr: bool = False, return_double: bool = False):

        if self.train_classifier_only and self.model.training:
            self.model.eval()

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        assert len(reps) == 2

        cated_input: Tensor = torch.cat((reps[0], reps[1]), dim=1)
        if self.train_classifier_only:
            cated_input = cated_input.data

        if return_dual:
            dual_input: Tensor = torch.cat((reps[1], reps[0]), dim=1)
            if self.train_classifier_only:
                dual_input = dual_input.data

        embeddings = self.embedding(cated_input)
        outputs = self.classifier(embeddings)

        if return_dual:
            embedding2 = self.embedding(dual_input)
            output2 = self.classifier(embedding2)

            if return_double:
                embeddings_db = self.embedding(cated_input)
                outputs_db = self.classifier(embeddings)
                embedding2_db = self.embedding(dual_input)
                output2_db = self.classifier(embedding2)

                return outputs, output2, outputs_db, output2_db

            if return_regr:
                s1 = self.regression(embeddings)
                s1 = torch.tanh(s1)
                s2 = self.regression(embedding2)
                s2 = torch.tanh(s2)
                return outputs, s1, output2, s2
            else:
                return outputs, output2
        else:
            if return_regr:
                s1 = self.regression(embeddings)
                s1 = torch.tanh(s1)
                return outputs, s1
            else:
                return outputs


logger = logging.getLogger(__name__)


class myEvaluator(BinaryClassificationEvaluator):
    def __init__(self, sentences1: List[str], sentences2: List[str], labels: List[int],
                 loss_model: CosineMimicLoss,
                 name: str = '', batch_size: int = 32, show_progress_bar: bool = False, write_csv: bool = False):

        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.loss_model = loss_model

        self.best_acc = float('-inf')

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        ce, acc = self.compute_ce_score(model)
        print('epoch{:d} step{:d}: ce_loss{:.2f}, acc{:.2f}'.format(epoch, steps, ce, acc*100))

        return acc*100-ce

    def compute_ce_score(self, model, return_probs=False):
        device = model._target_device
        sentences = list(set(self.sentences1 + self.sentences2))
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar,
                                      convert_to_numpy=True)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in self.sentences1]
        embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        cated_input = list()
        for emb1, emb2 in zip(embeddings1, embeddings2):
            cated_input.append(np.concatenate((emb1, emb2), axis=-1))
        cated_input = np.stack(cated_input)
        labels = np.asarray(self.labels)

        _ce_list = list()
        _pred_list = list()
        if return_probs:
            _probs_list = list()

        self.loss_model.eval()
        with torch.no_grad():
            for start_index in trange(0, len(cated_input), self.batch_size, desc="Batches",
                                      disable=not self.show_progress_bar):
                _batch = cated_input[start_index:start_index + self.batch_size]
                _label = labels[start_index:start_index + self.batch_size]
                _batch_tensor = torch.from_numpy(_batch).to(device)
                _label_tensor = torch.from_numpy(_label).to(device)

                _embedding_tensor = self.loss_model.embedding(_batch_tensor)
                _logits_tensor = self.loss_model.classifier(_embedding_tensor)

                if return_probs:
                    _probs_tensor = F.softmax(_logits_tensor, dim=-1)
                    _probs_list.append(_probs_tensor.detach().cpu().numpy())

                _pred_tensor = torch.argmax(_logits_tensor, dim=-1)
                _pred_list.append(_pred_tensor.detach().cpu().numpy())
                _ce_scores = F.cross_entropy(_logits_tensor, _label_tensor, reduction='none')
                _ce_list.append(_ce_scores.detach().cpu().numpy())
        _ce_list = np.concatenate(_ce_list, axis=-1)
        _pred_list = np.concatenate(_pred_list, axis=-1)
        if return_probs:
            _probs_list = np.concatenate(_probs_list, axis=-1)


        #'''
        uni_lb = np.unique(labels)
        lb_cnt = [np.sum(labels==lb) for lb in uni_lb]
        max_cnt = np.max(lb_cnt)
        w_lb = np.ones_like(labels, dtype=np.float32)
        for lb in uni_lb:
            idx = labels==lb
            w_lb[idx] = max_cnt / lb_cnt[lb]
        ce = np.sum(_ce_list * w_lb) / np.sum(w_lb)
        acc = np.sum((_pred_list == labels) * w_lb) / np.sum(w_lb)
        #'''

        # ce = np.mean(_ce_list)
        # acc = np.sum(_pred_list == labels) / len(labels)

        if return_probs:
            return float(ce), float(acc), _probs_list, _pred_list, labels

        return float(ce), float(acc)


def get_callback_save_fn(loss_model, outpath, demo_fn=None, seed=None):
    folder, fn = os.path.split(outpath)
    savepath = os.path.join(folder, 'model_part2.pt')
    loss_model.best_score = float('-inf')

    def _callback(score, epoch, steps):
        if score > loss_model.best_score:
            loss_model.best_score = score
            torch.save(loss_model, savepath)

            if score > 85:
                storepath, ext = os.path.splitext(savepath)
                fo, fn = os.path.split(storepath)
                fo = os.path.join(fo,'new')
                if not os.path.exists(fo):
                    os.makedirs(fo)
                fn += '_{:.2f}_{}.pt'.format(score, seed);
                storepath = os.path.join(fo,fn)
                torch.save(loss_model, storepath)
                print('store loss_model to', storepath)

            print('update best_score to', loss_model.best_score, 'save loss_model to', savepath)
            if demo_fn:
                demo_fn()

    return _callback


if __name__ == '__main__':
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    feature_dim = model.get_sentence_embedding_dimension()
    train_loss = CosineMimicLoss(model, feature_dim=feature_dim)

    train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0),
                      InputExample(texts=['Another pair', 'Unrelated sentence'], label=2)]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    train_loss.set_train_cosine()
    myevaluator = myEvaluator(['My first sentence'], ['My second sentence'], [0], loss_model=train_loss, batch_size=16)

    outpath = '../data/model_part1.pt'
    callback_fn = get_callback_save_fn(train_loss, outpath=outpath)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=0, output_path='all_part1.pt',
              evaluator=myevaluator, evaluation_steps=1, callback=callback_fn)
