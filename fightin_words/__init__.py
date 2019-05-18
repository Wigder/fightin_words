import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def weighted_log_odds_dirichlet(document1, document2, ngram=1, prior=.01, cv=None):
    """
    Arguments:
    - document1, document2; a list of strings from each sample.
    - ngram; an int describing up to what n gram you want to consider (1 is unigrams,
        2 is bigrams + unigrams, etc). Ignored if a custom CountVectorizer is passed.
    - prior; either a float describing a uniform prior, or a vector describing a prior
        over vocabulary items. If a predefined vocabulary is being used, make sure to
        specify that when making your CountVectorizer object.
    - cv; an sklearn.feature_extraction.text.CountVectorizer object, if necessary.

    Returns:
    - A list of length equal to the amount of vocabulary, where each entry is an (n-gram, zscore) tuple.
    """
    if cv is None and type(prior) is not float:
        raise TypeError("If using a non-uniform prior, a CountVectorizer object with the "
                        "vocabulary parameter set needs to be passed as the cv argument.")
    if cv is None:
        cv = CountVectorizer(decode_error='ignore', min_df=10, max_df=.5, ngram_range=(1, ngram), binary=False,
                             max_features=15000)
    counts_mat = cv.fit_transform(document1 + document2).toarray()
    vocab_size = len(cv.vocabulary_)
    if type(prior) is float:
        priors = np.array([prior] * vocab_size)
    else:
        priors = prior
    z_scores = np.empty(np.array(priors).shape[0])
    count_matrix = np.empty([2, vocab_size], dtype=np.float32)
    count_matrix[0, :] = np.sum(counts_mat[:len(document1), :], axis=0)
    count_matrix[1, :] = np.sum(counts_mat[len(document1):, :], axis=0)
    a0 = np.sum(priors)
    n1 = 1. * np.sum(count_matrix[0, :])
    n2 = 1. * np.sum(count_matrix[1, :])
    for i in range(vocab_size):
        term1 = np.log((count_matrix[0, i] + priors[i]) / (n1 + a0 - count_matrix[0, i] - priors[i]))
        term2 = np.log((count_matrix[1, i] + priors[i]) / (n2 + a0 - count_matrix[1, i] - priors[i]))
        delta = term1 - term2
        var = 1. / (count_matrix[0, i] + priors[i]) + 1. / (count_matrix[1, i] + priors[i])
        z_scores[i] = delta / np.sqrt(var)
    index_to_term = {v: k for k, v in cv.vocabulary_.items()}
    sorted_indices = np.argsort(z_scores)
    out = []
    for i in sorted_indices:
        out.append((index_to_term[i], z_scores[i]))

    return out
