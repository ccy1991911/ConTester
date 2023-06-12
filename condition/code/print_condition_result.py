import pickle
import NLP.data

dic = {}
with open('../data/condition_result_for_sentences.pickle', 'rb') as f:
    dic = pickle.load(f)

for md5 in dic:
    sent_text = dic[md5]['sent_text']
    mini_tree_set = dic[md5]['mini_tree_set']
    for x in mini_tree_set:
        conditions = x.conditions
        result = x.result
        probability = x.probability

        if result != None:
            print('>>', sent_text)
            print(conditions, '---->', result)
