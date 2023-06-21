from styleformer import Styleformer
#import NLP.change_tense

transformer_active_2_passive = Styleformer(style = 2)
transformer_passive_2_active = Styleformer(style = 3)

def active_2_passive(text):

    return transformer_active_2_passive.transfer(text)


def passive_2_active(text):

    return transformer_passive_2_active.transfer(text)


def present_2_past(text):

    return NLP.change_tense.change_tense(text, 'past')


def past_2_present(text):

    return NLP.change_tense.change_tense(text, 'present')


if __name__ == '__main__':

    text = 'The UE has started the attach procedure.'
    print(present_2_past(text))
    print(past_2_present(text))


