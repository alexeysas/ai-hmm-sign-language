import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # iterate through all test series
    for _, value in test_set.get_all_Xlengths().items():
        sequences = value[0]
        lengths = value[1]

        item_probabilities = {}

        # iterate through all words
        for word, model in models.items():
            try:
                # calculate prediction score
                res = model.score(sequences, lengths)
            except BaseException:
                res = float("-inf")

            item_probabilities[word] = res

        # append prediction scores and best guess to the resulting set.
        probabilities.append(item_probabilities)
        guesses.append(max(item_probabilities, key=item_probabilities.get))

    return probabilities, guesses
