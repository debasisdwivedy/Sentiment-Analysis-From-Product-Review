from src.BagOfWords import createBOW
from src.BagOfWords import getBOW
from src.BagOfWords import vectorize_without_Lexicon
from src.BagOfWords import vectorize_with_Lexicon
from src.BagOfWords import readPolarityFile
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    """POS Tagging"""
    review_BOW, review_Rating = createBOW('Product_Review.csv')
    Review_train_BOW, Review_test_BOW = train_test_split(review_BOW, train_size=0.75)
    Review_train_Rating, Review_test_Rating = train_test_split(review_Rating, train_size=0.75)

    feature_count = 3500
    # feature_count = None

    '''Without Polarity Lexicon Check'''

    ls = getBOW(Review_train_BOW, feature_count)
    vectorize_without_Lexicon("Train_TIMBL_WO_POL", Review_train_BOW, Review_train_Rating, ls)
    vectorize_without_Lexicon("Test_TIMBL_WO_POL", Review_test_BOW, Review_test_Rating, ls)

    '''With Polarity Lexicon Check'''

    polarity_dict = readPolarityFile('subjclueslen1-HLTEMNLP05.tff')
    ls = getBOW(Review_train_BOW, feature_count)
    vectorize_with_Lexicon("Train_TIMBL_W_POL", Review_train_BOW, Review_train_Rating, ls, polarity_dict)
    vectorize_with_Lexicon("Test_TIMBL_W_POL", Review_test_BOW, Review_test_Rating, ls, polarity_dict)
