import logging
from gensim.models import Doc2Vec
import re
import numpy as np
from nltk.corpus import stopwords
from gensim.models.doc2vec import LabeledSentence
from src.FetchData import readCSV
from sklearn.cross_validation import train_test_split
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

def getFeatureVecs(reviews, model, num_features):
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    counter = -1

    for review in reviews:
        counter += 1
        try:
            reviewFeatureVecs[counter] = np.array(model[review.labels[0]]).reshape((1, num_features))
        except:
            continue
    return reviewFeatureVecs

def getNegetors():
    return ['no','not','neither','never','none','nobody','nor','nothing','nowhere','hardly','rarely','scarcely','seldom','cannot']


def review_to_wordlist(review, remove_stopwords=False):
    review_text = re.sub('n\'t', ' not', review)

    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops or w in getNegetors()]
    return (words)


def getCleanLabeledReviews(reviews):
    clean_reviews = []
    for review in reviews["Review"]:
        clean_reviews.append(review_to_wordlist(review,True))

    labelized = []
    for i, id_label in enumerate(reviews["ID"]):
        labelized.append(LabeledSentence(clean_reviews[i], [id_label]))
    return labelized

def createModel_doc2Vec(Review_train,Review_test):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    # Set values for various parameters
    num_features = 5000  # Word vector dimensionality
    min_word_count = 1  # Minimum word count, if bigger, some sentences may be missing
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    model_dm = Doc2Vec(min_count=min_word_count, window=context, size=num_features, \
                       sample=downsampling, workers=num_workers)
    model_dbow = Doc2Vec(min_count=min_word_count, window=context, size=num_features,
                         sample=downsampling, workers=num_workers, dm=0)

    train_reviews = getCleanLabeledReviews(Review_train)
    test_reviews = getCleanLabeledReviews(Review_test)



    all_reviews=[]
    all_reviews+=train_reviews
    all_reviews+=test_reviews
    model_dm.build_vocab(all_reviews)
    model_dbow.build_vocab(all_reviews)


    for epoch in range(10):
        np.random.shuffle(all_reviews)
        model_dm.train(all_reviews)
        model_dbow.train(all_reviews)

    model_name = "Model_Doc2Vec"
    model_dbow_name="Model_Doc2Vec_BOW"
    model_dm.save(model_name)
    model_dbow.save(model_dbow_name)

def printResult(Review_test,result):
    i=0
    correct=0
    true_positive=0
    false_positive=0
    false_negetive=0
    for status in Review_test["Rating"]:
        if(status == result[i]):
            correct+=1
        if(status== 'Positive' and result[i]== 'Positive'):
            true_positive+=1

        if (status != 'Negative' and result[i] == 'Negative'):
            false_negetive += 1

        if (status != 'Positive' and result[i] == 'Positive'):
            false_positive += 1
        i+=1

    #Accuracy = Num. of Correct Queries / Total Num. of Queries
    print("Accuracy =",correct/i)

    #Precision = True Positive / ( True Positive + False Positive)
    print("Precision =", true_positive / (true_positive+false_positive))

    #Recall = Recall = True Positive / ( True Positive + False Negative
    print("Recall =", true_positive / (true_positive+false_negetive))

    # F1 = 2 * (Precision * Recall) / (Precision + Recall)

if __name__ == '__main__':
    review=readCSV('Product_Review.csv')

    train, test = train_test_split(review, train_size=0.7)

    clean_train_reviews = []
    for review in train['Review']:
        clean_train_reviews.append(" ".join(review_to_wordlist(review)))

    clean_test_reviews = []
    for review in test['Review']:
        clean_test_reviews.append(" ".join(review_to_wordlist(review)))

    createModel_doc2Vec(train,test)

    train_reviews = getCleanLabeledReviews(train)
    test_reviews = getCleanLabeledReviews(test)

    print("Creating the bag of words...\n")

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), sublinear_tf=True)

    X_train_bow = vectorizer.fit_transform(clean_train_reviews)
    X_test_bow = vectorizer.transform(clean_test_reviews)

    print('Loading doc2vec model..\n')

    model_dm_name = "Model_Doc2Vec"
    model_dbow_name = "Model_Doc2Vec_BOW"

    model_dm = Doc2Vec.load(model_dm_name)
    model_dbow = Doc2Vec.load(model_dbow_name)

    print("Creating the d2v vectors...\n")

    X_train_d2v_dm = getFeatureVecs(train_reviews, model_dm, 5000)
    X_train_d2v_dbow = getFeatureVecs(train_reviews, model_dbow, 5000)
    X_train_d2v = np.hstack((X_train_d2v_dm, X_train_d2v_dbow))

    X_test_d2v_dm = getFeatureVecs(test_reviews, model_dm, 5000)
    X_test_d2v_dbow = getFeatureVecs(test_reviews, model_dbow, 5000)
    X_test_d2v = np.hstack((X_test_d2v_dm, X_test_d2v_dbow))

    print("Combing the bag of words and the d2v vectors...\n")

    X_train_bdv = hstack([X_train_bow, X_train_d2v])
    X_test_bdv = hstack([X_test_bow, X_test_d2v])

    print("Checking the dimension of training vectors")

    print('D2V', X_train_d2v.shape)

    print('BoW-D2V', X_train_bdv.shape)

    y_train = train['Rating']

    clf = LogisticRegression(class_weight="auto")

    print("Predicting with Bag-of-words model and Doc2Vec model...\n")

    clf.fit(X_train_bdv, y_train)

    y_bdv=clf.predict(X_test_bdv)

    output = pd.DataFrame(data={"id": test["ID"], "Result": y_bdv,
                                "Actual_Result": test["Rating"]})
    output.to_csv("Doc2Vecor.csv", index=False, quoting=3)

    print("WORD 2 VECTOR + BAG OF WORDS")
    printResult(test, y_bdv)
