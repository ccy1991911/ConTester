import copy
import TD.utils
import NLP.allennlp
import NLP.stanza

class sentence():

    def __init__(self, sent_text):

        self.sent_text = sent_text
        self.tokens = TD.utils.my_tokenizer(sent_text)
        self.sent_text, self.tokens = TD.utils.exchange_nn_with_ving(self.sent_text, self.tokens)


class mini_tree():

    def __init__(self, conditions, result, probability):

        # conditions: {xxx, xxx, xxx, ...} (and)
        #              xxx = (text, introducer_word)
        # result: str
        # probability

        self.conditions = copy.deepcopy(conditions)
        self.result = result

        if type(probability) == type('string'):
            if probability in ['and', 'but']:
                self.probability = 1
            elif probability in ['or']:
                self.probability = 0.5
        else:
            self.probability = probability

