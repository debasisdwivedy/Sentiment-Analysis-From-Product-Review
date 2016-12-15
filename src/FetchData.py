import pandas as pd
from sklearn.cross_validation import train_test_split
import re
from nltk.corpus import stopwords
import nltk
import csv

def splitData(X,Y,split):
    if(split==None):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=split)
    return X_train,X_test,Y_train,Y_test

def readCSV(filename):
    reviewers = pd.read_csv(open(filename,'rU',encoding='iso-8859-1'))
    # Drop rows with missing data
    reviewers = reviewers[~reviewers.Review.isnull()]
    reviewers = reviewers[reviewers.Rating != 'none']
    return reviewers

def optimizeFeatures(quotes,remove_stopwords=False):
    reviewers = re.sub("[^a-zA-Z]", " ", quotes)
    words = reviewers.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words

def review_to_sentences( review, remove_stopwords=False ):

    raw_sentences = nltk.tokenize.sent_tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( optimizeFeatures(raw_sentence,remove_stopwords ))
    return sentences

def create_CSV(input_file,output_file):
    word_dict = dict()
    word_dict["rating"] = []
    word_dict["product_ID"] = []
    word_dict["helpfulness"] = []
    word_dict["ID"] = []
    word_dict["review_by"] = []
    word_dict["title"] = []
    word_dict["review_time"] = []
    word_dict["review"] = []
    with open(input_file) as f:
        contents = f.readlines()
        for content in contents:
            data = content.split(":")
            try:
                value=data[1].rstrip('\n')
                if(data[0]=='rating'):
                    rate=int(value.strip()[0])
                    if(rate<3):
                        value='Negative'
                    elif(rate>3):
                        value='Positive'
                    else:
                        value='Neutral'
                word_dict[data[0]].append(value)
            except:
                pass

    output = pd.DataFrame(data={"Rating": word_dict["rating"], "Product_ID": word_dict["product_ID"],
                                "Helpfulness": word_dict["helpfulness"], "ID": word_dict["ID"],
                                "Review_by": word_dict["review_by"], "Title": word_dict["title"],
                                "Review_time": word_dict["review_time"], "Review": word_dict["review"]})
    output.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)




if __name__ == '__main__':
    txt_file = "Dataset/amazon_Micro.txt"
    csv_file = "Product_Review.csv"
    create_CSV(txt_file,csv_file)