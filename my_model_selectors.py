import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def build_model(self, n_states):
        '''
        Abstract method to fit model and calculate score,
        so it can be compared with other models by 'select' method
        '''
        raise NotImplementedError

    def select(self):
        '''
        Enumerate possible parametrs (number of states)
        and select model with minimum error
        '''
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        #set default model in case we always fail
        best_num_components = self.n_constant
        best_model = self.base_model(best_num_components)
        best_score = float("inf")

        for n_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model, score = self.build_model(n_states)
                if score < best_score:
                    best_score = score
                    best_model = model
                    best_num_components = n_states
            except BaseException:
                print("Failed to build model woth n_states={}".format(n_states))

        return best_model

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def build_model(self, n_states):
        '''
        Select best model based on BIC:
        https://rdrr.io/cran/HMMpa/man/AIC_HMM.html

        BIC = -2 logL + p log(N)
        N is the number of data points
        p is the number of parameters
        L is the likelihood of the fitted model

        '''

        # train model
        model = self.base_model(n_states)
        model = model.fit(self.X, self.lengths)

        # calculate log-liklyhood for the model
        log_l_score = model.score(self.X, self.lengths)

        # calculate N
        N = np.mean(self.lengths)
        # not exactly sure if we need mean or sum.
        # probably as soon as we are consistent with this either might work

        # calculate p
        # to calculate p we need to sum up all parameters for the model which are trainables
        # according to the links trainables are: model.startprob_,
        # model.transmat_, model.covars_, model.means
        # http://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn-hmm
        # https://github.com/hmmlearn/hmmlearn/blob/master/hmmlearn/hmm.py
        # as sums of rows are equal to 1 so last parameter is calculated
        # and can not be considered as trainable
        p = np.size(model.transmat_) - np.size(model.transmat_, 0)

        # again sum should be equal to 1 so substracting one value
        p += np.size(model.startprob_) - 1

        # for the covariance_type="diag" only diagonal elements should be considered
        # so instead of np.size(model.covars_) going to use np.size(model.means_)
        # as using np.size(model.covars_) will lead to incorrect results.
        p += np.size(model.means_) + np.size(model.means_)

        # combine score
        score = -2 * log_l_score + p * np.log(N)

        return model, score


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    '''select best model based on average log Likelihood of cross-validation folds

    '''

    def build_model(self, n_states):
        '''
        select best model based on average log Likelihood of cross-validation folds
        '''

        split_method = KFold()
        scores = []

        # enumerate index sets
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

            # prepare training and test data
            train_set_data, train_set_lengths = combine_sequences(cv_train_idx, self.sequences)
            test_set_data, test_set_lengths = combine_sequences(cv_test_idx, self.sequences)

            # train model
            model = self.base_model(n_states)
            model = model.fit(train_set_data, train_set_lengths)

            # calculate log-liklyhood for the model
            log_l_score = model.score(test_set_data, test_set_lengths)

            # collect score to 
            scores.append(log_l_score)

        print(np.mean(scores))
        return model, np.mean(scores)


