from src.FetchData import readCSV
from src.BagOfWords import vectorize
from src.OptimizedVectors import optimize
from src.FetchData import splitData
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

def trainSystem_default(critics):
    X, Y = vectorize(critics)
    clf = MultinomialNB()
    Default_Naive_Bayesian_Classifier(X,Y, clf)


def Default_Naive_Bayesian_Classifier(X,Y,clf):
    X_train, X_test, Y_train, Y_test = splitData(X, Y,0.75)
    clf.fit(X_train, Y_train)
    ypred = clf.predict(X_test)

    print('Default_Naive_Bayesian Accuracy:', round(clf.score(X_test, Y_test), 2))
    print('Default_Naive_Bayesian Precision:', round(precision_score(Y_test, ypred), 2))
    print('Default_Naive_Bayesian Recall:', round(recall_score(Y_test, ypred), 2))
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)


def trainSystem_optimized(critics,best_min_df,best_alpha):
    X, Y = vectorize(critics, CountVectorizer(min_df=best_min_df))
    clf_opt = MultinomialNB(alpha=best_alpha)
    Optimized_Naive_Bayesian_Classifier(X,Y,clf_opt)

def Optimized_Naive_Bayesian_Classifier(X,Y,clf_opt):
    scores = []
    for train, test in KFold(Y.size, 5):
        Xtrain, Xtest, Ytrain, Ytest = X[train], X[test], Y[train], Y[test]
        clf_opt.fit(Xtrain, Ytrain)
        ypred = clf_opt.predict(Xtest)
        accuracy = accuracy_score(Ytest, ypred)
        precision = precision_score(Ytest, ypred)
        recall = recall_score(Ytest, ypred)
        scores.append((accuracy, precision, recall))


    print("Optimized_Naive_Bayesian Accuracy: " + str(round(np.mean([x[0] for x in scores]), 2)))
    print("Optimized_Naive_Bayesian Precision: " + str(round(np.mean([x[1] for x in scores]), 2)))
    print("Optimized_Naive_Bayesian Recall: " + str(round(np.mean([x[2] for x in scores]), 2)))
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)

if __name__ == '__main__':
    review = readCSV('Product_Review.csv')

    num_reviewers = review[['Review_by']].drop_duplicates().shape[0]
    num_products = review[['Product_ID']].drop_duplicates().shape[0]
    num_reviews = review.shape[0]

    print('number of reviewers:', num_reviewers)
    print('number of products:', num_products)
    print('number of reviews:', num_reviews)


    #Default Result
    trainSystem_default(review);


    #Optimized Results
    best_alpha, best_min_df=optimize(review)
    trainSystem_optimized(review,best_min_df,best_alpha)
