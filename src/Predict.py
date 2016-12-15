from src.FetchData import create_CSV
from src.FetchData import readCSV
from src.NaiveBayes import trainSystem_default
from src.NaiveBayes import trainSystem_optimized
from src.OptimizedVectors import optimize
from src.SVM import classify_support_vector_machine
from sklearn.cross_validation import train_test_split
from src.Word2Vector import review_to_wordlist
from src.Word2Vector import createModel_word2Vec
from src.Word2Vector import getAvgFeatureVecs
from src.Word2Vector import printResult
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from sklearn.preprocessing import scale
import pandas as pd
from gensim.models import Doc2Vec
import numpy as np
from src.Doc2Vector import getFeatureVecs
from src.Doc2Vector import getCleanLabeledReviews
from src.Doc2Vector import createModel_doc2Vec
from src.BagOfWords import BOW_Scikit_learn
from src.BagOfWords import optimizeFeatures
from src.BagOfWords import RandomForestImpl

if __name__ == '__main__':

    ############ Create File #####################################

    txt_file = "Dataset/amazon_Micro.txt"
    csv_file = "Product_Review.csv"
    create_CSV(txt_file, csv_file)

    ############### Naive Bayesian #################################

    review = readCSV('Product_Review.csv')

    num_critics = review[['Review_by']].drop_duplicates().shape[0]
    num_products = review[['Product_ID']].drop_duplicates().shape[0]
    num_reviews = review.shape[0]

    print('number of reviewers:', num_critics)
    print('number of products:', num_products)
    print('number of reviews:', num_reviews)

    # Default Result
    trainSystem_default(review)

    # Optimized Results
    best_alpha, best_min_df = optimize(review)
    trainSystem_optimized(review, best_min_df, best_alpha)
    # calibration_plot(clf, X, Y)


    ################ SVM ###########################################

    classifier_linear = classify_support_vector_machine(best_min_df,review)

    ################ Word 2 Vector ###########################################

    Review_train, Review_test = train_test_split(review, train_size=0.75)

    # clean_train_reviews = getCleanReviews(Review_train)
    # clean_test_reviews = getCleanReviews(Review_test)

    clean_train_reviews = []
    for review in Review_train['Review']:
        clean_train_reviews.append(" ".join(review_to_wordlist(review)))

    clean_test_reviews = []
    for review in Review_test['Review']:
        clean_test_reviews.append(" ".join(review_to_wordlist(review)))

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), sublinear_tf=True)

    X_train_bow = vectorizer.fit_transform(clean_train_reviews)
    X_test_bow = vectorizer.transform(clean_test_reviews)

    model_final = createModel_word2Vec(clean_train_reviews)
    print('Loading word2vec model..\n')

    model = Word2Vec.load(model_final)

    print("Creating the w2v vectors...\n")

    X_train_w2v = scale(getAvgFeatureVecs(clean_train_reviews, model, 5000))
    X_test_w2v = scale(getAvgFeatureVecs(clean_test_reviews, model, 5000))

    print("Combing the bag of words and the w2v vectors...\n")

    X_train_bwv = hstack([X_train_bow, X_train_w2v])
    X_test_bwv = hstack([X_test_bow, X_test_w2v])

    print("Checking the dimension of training vectors")

    print('W2V', X_train_w2v.shape)

    print('BoW-W2V', X_train_bwv.shape)

    y_train = Review_train['Rating']

    clf = LogisticRegression(class_weight="auto")

    print("Predicting with Bag-of-words model and Word2Vec model...\n")

    clf.fit(X_train_bwv, y_train)
    y_bwv = clf.predict(X_test_bwv)

    output = pd.DataFrame(data={"id": Review_test["ID"], "Result": y_bwv,
                                "Actual_Result": Review_test["Rating"]})
    output.to_csv("BagOfCentroids.csv", index=False, quoting=3)

    print("WORD 2 VECTOR + BAG OF WORDS")
    printResult(Review_test, y_bwv)

    ##################### Doc 2 Vector ######################################

    review = readCSV('Product_Review.csv')
    # review = pd.read_csv('Product_Review.csv', header=0, delimiter="\t", quoting=3, encoding='iso-8859-1')
    train, test = train_test_split(review, train_size=0.7)

    clean_train_reviews = []
    for review in train['Review']:
        clean_train_reviews.append(" ".join(review_to_wordlist(review)))

    clean_test_reviews = []
    for review in test['Review']:
        clean_test_reviews.append(" ".join(review_to_wordlist(review)))

    createModel_doc2Vec(train, test)

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

    y_bdv = clf.predict(X_test_bdv)

    output = pd.DataFrame(data={"id": test["ID"], "Result": y_bdv,
                                "Actual_Result": test["Rating"]})
    output.to_csv("Doc2Vecor.csv", index=False, quoting=3)

    print("WORD 2 VECTOR + BAG OF WORDS")
    printResult(test, y_bdv)

    ###################### BOW + Random Forest ##########################################
    clean_train_reviews = []
    for review in Review_train['Review']:
        clean_train_reviews.append(optimizeFeatures(review, True))

    clean_test_reviews = []
    for review in Review_test['Review']:
        clean_test_reviews.append(optimizeFeatures(review, True))

    vectorizer, train_data_features = BOW_Scikit_learn(clean_train_reviews)
    RandomForestImpl(vectorizer, train_data_features, clean_test_reviews)

