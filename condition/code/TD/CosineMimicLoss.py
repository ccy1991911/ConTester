from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
from typing import Iterable, Dict, List
from torch import Tensor
import numpy as np
from tqdm.autonotebook import trange
import logging


class CosineMimicLoss(torch.nn.Module):
    def __init__(self, model: SentenceTransformer, feature_dim: int):
        super(CosineMimicLoss, self).__init__()
        self.model = model
        self.device = model.device
        self.feature_dim = feature_dim
        self.classifier = self.get_classifier(feature_dims=[512, 256])
        self.mode = 0
        self.train_classifier_only = False

    def get_classifier(self, feature_dims: Iterable[int]):
        list_layer = list()
        last_dim = self.feature_dim * 2
        for i, dim in enumerate(feature_dims):
            fn = torch.nn.Linear(in_features=last_dim, out_features=dim)
            list_layer.append(('fn{:d}'.format(i + 1), fn))
            relu = torch.nn.ReLU()
            list_layer.append(('relu{:d}'.format(i + 1), relu))
            last_dim = dim
        fn = torch.nn.Linear(in_features=last_dim, out_features=4)
        list_layer.append(('fn{:d}'.format(len(feature_dims) + 1), fn))
        _module = torch.nn.Sequential(OrderedDict(list_layer))
        _module = _module.to(self.device)
        return _module

    def set_train_classifier(self):
        self.mode = 1

    def set_train_all(self):
        self.mode = 2

    def set_predict(self):
        self.mode = 0

    def set_train_classifier_only(self):
        self.train_classifier_only = True
        self.model.eval()

    def reset_train_classifier_only(self):
        self.train_classifier_only = False

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor = None):
        return_reps = False
        if self.mode == 1:
            return_reps = True
        outputs = self._forward(sentence_features, return_reps)

        if self.mode == 1:
            outputs, reps0, reps1 = outputs
            cosine_value: Tensor = torch.cosine_similarity(reps0, reps1)
            cosine_value = cosine_value / 2.0 + 0.5

            targets = torch.zeros_like(outputs)
            targets[:, 0] = 1 - cosine_value
            targets[:, 1] = cosine_value

            loss = torch.nn.functional.cross_entropy(outputs, targets.data)
            return loss
        elif self.mode == 2:
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            return loss
        elif self.mode == 0:
            outputs = F.softmax(outputs, dim=-1)
            return outputs
        else:
            raise NotImplementedError

    def _forward(self, sentence_features: Iterable[Dict[str, Tensor]], return_reps: bool = False):
        if self.train_classifier_only and self.model.training:
            self.model.eval()

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        assert len(reps) == 2

        cated_input: Tensor = torch.cat((reps[0], reps[1]), dim=1)

        if self.train_classifier_only:
            cated_input = cated_input.data

        outputs = self.classifier(cated_input)
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

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        ce, acc = self.compute_ce_score(model)
        print('epoch{:d} step{:d}: ce_loss{:.2f}, acc{:.2f}'.format(epoch, steps, ce, acc*100))
        return -ce

    def compute_ce_score(self, model):
        device = model.device
        sentences = list(set(self.sentences1 + self.sentences2))
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

        if self.loss_model.device != device:
            self.loss_model.to(device)
            self.loss_model.device = device

        _ce_list = list()
        _pred_list = list()
        with torch.no_grad():
            for start_index in trange(0, len(cated_input), self.batch_size, desc="Batches",
                                      disable=not self.show_progress_bar):
                _batch = cated_input[start_index:start_index + self.batch_size]
                _label = labels[start_index:start_index + self.batch_size]
                _batch_tensor = torch.from_numpy(_batch).to(device)
                _label_tensor = torch.from_numpy(_label).to(device)
                _logits_tensor = self.loss_model.classifier(_batch_tensor)
                _pred_tensor = torch.argmax(_logits_tensor, dim=-1)
                _pred_list.append(_pred_tensor.detach().cpu().numpy())
                _ce_scores = F.cross_entropy(_logits_tensor, _label_tensor, reduction='none')
                _ce_list.append(_ce_scores.detach().cpu().numpy())
        _ce_list = np.concatenate(_ce_list, axis=-1)
        _pred_list = np.concatenate(_pred_list, axis=-1)
        ce = np.mean(_ce_list)
        acc = np.sum(_pred_list == labels) / len(labels)

        return float(ce), float(acc)


def get_callback_save_fn(loss_model, outpath):
    folder, fn = os.path.split(outpath)
    savepath = os.path.join(folder, 'model_part2.pt')
    loss_model.best_score = float('-inf')

    def _callback(score, epoch, steps):
        if score > loss_model.best_score:
            loss_model.best_score = score
            torch.save(loss_model, savepath)
            print('update best_score to', loss_model.best_score, 'save loss_model to', savepath)

    return _callback


if __name__ == '__main__':
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    feature_dim = model.get_sentence_embedding_dimension()
    train_loss = CosineMimicLoss(model, feature_dim=feature_dim)

    train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0),
                      InputExample(texts=['Another pair', 'Unrelated sentence'], label=2)]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    train_loss.set_train_classifier()
    myevaluator = myEvaluator(['My first sentence'], ['My second sentence'], [0], loss_model=train_loss, batch_size=16)

    outpath = '../data/model_part1.pt'
    callback_fn = get_callback_save_fn(train_loss, outpath=outpath)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=0, output_path='all_part1.pt',
              evaluator=myevaluator, evaluation_steps=1, callback=callback_fn)
