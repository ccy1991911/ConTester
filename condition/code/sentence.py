import TD.utils
import NLP.allennlp
import NLP.stanza
import NLP.data
import copy

def analyze_sent_on_result_verb(sent):

    srl_result = NLP.allennlp.get_srl_result(sent.tokens)
    srl_item_on_level_1, conj_word = NLP.allennlp.get_srl_item_on_level_1(sent, srl_result)

    mini_tree_set = set()

    for item in srl_item_on_level_1:
        conditions, result = NLP.allennlp.get_condition_and_result(sent, item)
        if result != None:
            mini_tree_set.add(NLP.data.mini_tree(conditions, result, conj_word))

    return mini_tree_set


def analyze_sent_on_condition_verb(mini_tree_set):

    extend_mini_tree_set = set()
    extend_flag = True

    while extend_flag == True:

        extend_flag = False
        extend_mini_tree_set = set()

        for mt in mini_tree_set:

            conditions = mt.conditions
            result = mt.result
            probability = mt.probability

            # choose one condition from conditions to extend
            to_extend_c_sent_text = None
            to_extend_c_introducer_word = None
            to_extend_c_sent = None
            to_extend_c_srl_item_on_level_1 = None
            to_extend_c_conj_word = None
            for c_sent_text, c_introducer_word in conditions:
                c_sent = NLP.data.sentence(c_sent_text)
                c_srl_result = NLP.allennlp.get_srl_result(c_sent.tokens)
                c_srl_item_on_level_1, c_conj_word = NLP.allennlp.get_srl_item_on_level_1(c_sent, c_srl_result)

                if len(c_srl_item_on_level_1) > 1:
                    to_extend_c_sent_text = c_sent_text
                    to_extend_c_introducer_word = c_introducer_word
                    to_extend_c_sent = c_sent
                    to_extend_c_srl_item_on_level_1 = c_srl_item_on_level_1
                    to_extend_c_conj_word = c_conj_word
                    break

            # extend
            if to_extend_c_sent_text == None:
                extend_mini_tree_set.add(copy.deepcopy(mt))
            else:
                extend_flag = True

                extend_conditions = []
                for c_sent_text, c_introducer_word in conditions:
                    if c_sent_text != to_extend_c_sent_text:
                        extend_conditions.append((c_sent_text, c_introducer_word))


                if to_extend_c_conj_word in ['and']:
                    for item in to_extend_c_srl_item_on_level_1:
                        extend_c_sent_text = NLP.allennlp.get_text(to_extend_c_sent, item, set())
                        extend_c_introducer_word = to_extend_c_introducer_word
                        extend_conditions.append((extend_c_sent_text, extend_c_introducer_word))
                    extend_mini_tree_set.add(NLP.data.mini_tree(extend_conditions, result, to_extend_c_conj_word if to_extend_c_conj_word in ['or'] else probability))
                elif to_extend_c_conj_word in ['or']:
                    for item in to_extend_c_srl_item_on_level_1:
                        extend_c_sent_text = NLP.allennlp.get_text(to_extend_c_sent, item, set())
                        extend_c_introducer_word = to_extend_c_introducer_word
                        extend_conditions.append((extend_c_sent_text, extend_c_introducer_word))
                        extend_mini_tree_set.add(NLP.data.mini_tree(extend_conditions, result, to_extend_c_conj_word if to_extend_c_conj_word in ['or'] else probability))
                        extend_conditions.pop()

        mini_tree_set = copy.deepcopy(extend_mini_tree_set)

    return mini_tree_set


def analyze_sent(sent_text = None):

    if sent_text == None:
        sent_text = 'Upon receipt of the ATTACH REJECT message, if the message is with EMM cause #22 or #23 or the message is not integrity protected, the UE shall delete the GUTI, send a response to the MME, or entering the state EMM-DERIGESTERED.'
        sent_text = 'If the attach attempt counter is equal to 5, the UE shall delete any GUTI, TAI list, last visited registered TAI, list of equivalent PLMNs and KSI.'
        sent_text = 'The MME initiates the NAS security mode control procedure by sending a SECURITY MODE COMMAND message to the UE and starting timer T3460.'
        sent_text = 'Secure exchange of NAS messages via a NAS signalling connection is usually established by the MME during the attach procedure by initiating a security mode control procedure.'

    print('\n>>', sent_text)

    sent = NLP.data.sentence(sent_text)
    mini_tree_set = analyze_sent_on_result_verb(sent)
    mini_tree_set = analyze_sent_on_condition_verb(mini_tree_set)
    output_mini_tree_set(mini_tree_set)

    return mini_tree_set





def test_all():

    file = open('../data/testing_sentences.txt')
    fileContent = file.readlines()
    file.close()

    for tmp in fileContent:
        if tmp.startswith('#'):
            continue
        if len(tmp.strip()) > 0:
            analyze_sent(tmp.strip())


def output_mini_tree_set(mini_tree_set):

    print('-->> result -->>')
    for x in mini_tree_set:
        print('- condition -')
        for c in x.conditions:
            print(c[1], ':', c[0])
        print('- result -')
        print(x.result)
        print('- probability - ')
        print(x.probability)


if __name__ == '__main__':

    test_all()
    #analyze_sent()
