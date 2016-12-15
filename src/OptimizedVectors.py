import numpy as np
from src.BagOfWords import vectorize
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold

"""Stop overfitting by using five fold cross validation."""

def probability(clf, x, y):

    prob = clf.predict_log_proba(x)

    positive = []
    negetive = []

    for x, p in enumerate(prob):

        if y[x] == 1:

            positive.append(p[1])

        elif y[x] == 0:

            negetive.append(p[0])

    L = sum(positive) + sum(negetive)

    return L

def updated_result(clf, x, y, score_func):
    result_obtained = 0
    cross_validate = 5
    for train, test in KFold(y.size, cross_validate):
        clf.fit(x[train], y[train])
        result_obtained += score_func(clf, x[test], y[test])
    return result_obtained / cross_validate

def optimize(reviewers):
    alphas = [.1, 1, 5, 10, 50]
    min_dfs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    # The best classifier for the alpha obtained
    best_alpha = None
    best_min_df = None
    max_prob = -np.inf

    for alpha in alphas:
        for min_df in min_dfs:
            vectorizer = CountVectorizer(min_df=min_df)
            X, Y = vectorize(reviewers, vectorizer)

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

            clf = MultinomialNB(alpha=alpha)

            this_prob = updated_result(clf, X_train, Y_train, probability)

            # Update probability
            if this_prob > max_prob:
                max_prob = this_prob
                best_alpha = alpha
                best_min_df = min_df


    print("best value of alpha obtained: %f" % best_alpha)
    print("bealue of min_df obtainedst v: %f" % best_min_df)

    return best_alpha,best_min_df

