import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from src.FetchData import readCSV
from sklearn.cross_validation import train_test_split

"""Build a bag-of-words training set for the review data"""

def vectorize(reviewers, vectorizer=None):
    X = []
    Y = []

    # if no vectorizer is passed to function, create one with min_df=0
    if vectorizer == None:
        vectorizer = CountVectorizer(min_df=0)

    # fit vectorizer to entire Review repertoire
    vectorizer.fit(reviewers.Review)
    # make bag of words from Review
    X = vectorizer.transform(reviewers.Review)

    for x, y in enumerate(reviewers['Review']):

        rating = reviewers['Rating'].iloc[x]
        if rating == 'Positive':
            val = 1
        else:
            val = 0

        Y.append(val)

    return X, np.asarray(Y)

def vectorize_with_Lexicon(filename,word_list_train, word_list_stance, total_words,polarity_dict):

    final_list=[]

    final_list.append(total_words+['Rating'])

    i = 0
    for sublist in word_list_train:
        ls_sub=[]
        counts = Counter(sublist)
        for word in total_words:
            if counts[word] is None:
                ls_sub.append(0)
            else:
                x = 1
                try:
                    x = polarity_dict[word]
                    ls_sub.append(x * counts[word])
                except KeyError:
                    ls_sub.append(counts[word])
                    pass


        ls_sub.append(word_list_stance[i])
        final_list.append(ls_sub)
        i += 1

    my_df = pd.DataFrame(final_list)
    my_df.to_csv("timbl-myfile-with-Lexicon/"+filename+'.csv', index=False, header=False)

def vectorize_without_Lexicon(filename,word_list_train, word_list_stance, total_words):
    # print(total_words)
    # print(clean_train_reviews)
    # clean_test_reviews = [item for sublist in word_list_test for item in sublist]

    final_list = []

    final_list.append(total_words + ['Rating'])

    i = 0
    for sublist in word_list_train:
        ls_sub = []
        counts = Counter(sublist)
        for word in total_words:
            if counts[word] is None:
                ls_sub.append(0)
            else:
                ls_sub.append(counts[word])

        ls_sub.append(word_list_stance[i])
        final_list.append(ls_sub)
        i += 1

    '''for ls in final_list:
        print(ls)'''

    my_df = pd.DataFrame(final_list)
    my_df.to_csv("timbl-myfile-without-Lexicon/" + filename + '.csv', index=False, header=False)

def createBOW(filename):
    list_words, list_polarity = POS_Tagging(filename)
    return list_words, list_polarity

def POS_Tagging(filename):
    list_words=[]
    list_polarity=[]
    for row in pd.read_csv(open(filename, 'rU', encoding='iso-8859-1'),chunksize=1):

        NN = []  # Noun
        VBP = []  # Verb
        JJ = []  # Adjective

        for item in row.Review:
            review = optimizeFeatures(item,True) # for non optimized features

        for item in row.Rating:
            rating=item

        NN.append([token for token, pos in nltk.pos_tag(nltk.word_tokenize(review)) if pos.startswith('N')])
        VBP.append([token for token, pos in nltk.pos_tag(nltk.word_tokenize(review)) if pos.startswith('V')])
        JJ.append([token for token, pos in nltk.pos_tag(nltk.word_tokenize(review)) if pos.startswith('J')])
        list_words.append([item for sublist in NN + VBP + JJ for item in sublist])
        list_polarity.append(rating)


    return list_words,list_polarity

def optimizeFeatures(item,remove_stopwords):
    letters_only = re.sub("[^a-zA-Z]", " ", item)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    if remove_stopwords:
        meaningful_words = [w for w in words if not w in stops]
        review = " ".join(meaningful_words)
    return review

def getBOW(word_list_train,feature_count):
    clean_train_reviews = [item for sublist in word_list_train for item in sublist]
    ls=[]
    ls.append(clean_train_reviews)
    if(feature_count is not None):
        ls=list(set([item for sublist in ls for item in sublist]))[:feature_count]
    else:
        ls = list(set([item for sublist in ls for item in sublist]))
    return ls

def readPolarityFile(filename):
    polarity_dict={}
    with open(filename) as f:
        content = f.readlines()
        for line in content:
            status = line.split()
            try:
                d = dict(s.split('=') for s in status)
                polarity=d["priorpolarity"]
                subject=d["type"]
                if subject == "strongsubj":
                    if polarity == "positive":
                        polarity_dict[d["word1"]] = 2
                    elif polarity == "negative":
                        polarity_dict[d["word1"]] = -2
                    elif polarity == "neutral":
                        polarity_dict[d["word1"]] = 0
                else:
                    if polarity == "positive":
                        polarity_dict[d["word1"]] = 1
                    elif polarity == "negative":
                        polarity_dict[d["word1"]] = -1
                    elif polarity == "neutral":
                        polarity_dict[d["word1"]] = 0
            except:
                pass

    return polarity_dict

def BOW_Scikit_learn(clean_train_reviews):
    print("Creating the bag of words...\n")

    vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    train_data_features = train_data_features.toarray()
    print(train_data_features.shape)
    vocab = vectorizer.get_feature_names()
    #print(vocab)

    return vectorizer,train_data_features

def RandomForestImpl(vectorizer,train_data_features,clean_test_reviews):
    print("Training the random forest...")
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train["Rating"])

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame(data={"id": test["ID"], "Result": result,
                                "Actual_Result": test["Rating"]})

    # Use pandas to write the comma-separated output file
    output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)
    print("RANDOM FOREST CLASSIFIER + BAG OF WORDS")
    printResult(test,result)

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
    review = readCSV('Product_Review.csv')
    train, test = train_test_split(review, train_size=0.75)

    clean_train_reviews = []
    for review in train['Review']:
        clean_train_reviews.append(optimizeFeatures(review,True))

    clean_test_reviews = []
    for review in test['Review']:
        clean_test_reviews.append(optimizeFeatures(review,True))

    vectorizer,train_data_features = BOW_Scikit_learn(clean_train_reviews)

    RandomForestImpl(vectorizer,train_data_features,clean_test_reviews)
