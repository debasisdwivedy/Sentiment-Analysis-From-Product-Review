from gensim.models import Word2Vec
from src.FetchData import readCSV
from src.FetchData import review_to_sentences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from sklearn.preprocessing import scale
import logging
from sklearn.cluster import KMeans
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import re
from nltk.corpus import stopwords

def createModel_word2Vec(sentences):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    # Set values for various parameters
    num_features = 5000  # Word vector dimensionality
    min_word_count = 5  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10 # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training model...")
    model = Word2Vec(sentences,workers=num_workers,size=num_features, min_count=min_word_count,window=context, sample=downsampling)
    model.init_sims(replace=True)
    model_name = "Model"
    model.save(model_name)
    return model_name

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

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0

    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])

    if nwords != 0:
        featureVec /= nwords
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    counter = 0

    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["Review"]:
        clean_reviews += review_to_sentences(review,True)
    return clean_reviews

def K_Means_Clustering(model):
    start = time.time()
    word_vectors = model.syn0
    num_clusters = int(word_vectors.shape[0] / 5)
    print(num_clusters)

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print("Time taken for K Means clustering: ", elapsed, "seconds.")
    word_centroid_map = dict(zip(model.index2word, idx))
    return word_centroid_map,num_clusters

def printCluster(word_centroid_map):
    for cluster in range(0, 10):
        #
        # Print the cluster number
        print("\nCluster %d" % cluster)
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in range(0, len(word_centroid_map.values())):
            v=list(word_centroid_map.values())
            k=list(word_centroid_map.keys())
            if (v[i] == cluster):
                words.append(k[i])
        print(words)

def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

def BOC(word_centroid_map,num_clusters,Review_train,Review_test):
    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros((Review_train['Review'].size, num_clusters),dtype="float32")

    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in Review_train['Review']:
        train_centroids[counter] = create_bag_of_centroids(review,word_centroid_map)
        counter += 1

    # Repeat for test reviews
    test_centroids = np.zeros((Review_test['Review'].size, num_clusters),dtype="float32")

    counter = 0
    for review in Review_test['Review']:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    return train_centroids,test_centroids

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
    reviewers=readCSV('Product_Review.csv')
    Review_train, Review_test=train_test_split(reviewers, train_size=0.75)

    #clean_train_reviews = getCleanReviews(Review_train)
    #clean_test_reviews = getCleanReviews(Review_test)

    clean_train_reviews = []
    for review in Review_train['Review']:
        clean_train_reviews.append(" ".join(review_to_wordlist(review)))

    clean_test_reviews = []
    for review in Review_test['Review']:
        clean_test_reviews.append(" ".join(review_to_wordlist(review)))


    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), sublinear_tf=True)

    X_train_bow = vectorizer.fit_transform(clean_train_reviews)
    X_test_bow = vectorizer.transform(clean_test_reviews)


    model_final=createModel_word2Vec(clean_train_reviews)
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
    output.to_csv("Word2Vecor.csv", index=False, quoting=3)

    print("WORD 2 VECTOR + BAG OF WORDS")
    printResult(Review_test, y_bwv)

