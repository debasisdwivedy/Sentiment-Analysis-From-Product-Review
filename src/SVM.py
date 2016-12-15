from src.BagOfWords import vectorize
from src.FetchData import splitData
from src.FetchData import readCSV
from sklearn import svm
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer


def classify_support_vector_machine(best_min_df,reviewers):
    scores = []
    vectorizer = CountVectorizer(min_df=best_min_df)
    x_axis,y_axis=vectorize(reviewers,vectorizer)
    x_axis_train, x_axis_test, y_axis_train, y_axis_test = splitData(x_axis, y_axis,0.75)
    for train, test in KFold(y_axis.size, 5):
        new_x_axis_train, new_x_axis_test, new_y_axis_train, new_y_axis_test = x_axis[train], x_axis[test], y_axis[train], y_axis[test]
        classifier_linear = svm.SVC(probability=True,kernel='linear')
        classifier_linear.fit(x_axis_train, y_axis_train)
        predicted_y = classifier_linear.predict(new_x_axis_test)
        accuracy = accuracy_score(new_y_axis_test, predicted_y)
        precision = precision_score(new_y_axis_test, predicted_y)
        recall = recall_score(new_y_axis_test, predicted_y)
        scores.append((accuracy, precision, recall))
    print("Accuracy obtained from SVM: " + str(round(np.mean([x[0] for x in scores]), 2)))
    print("Precision obtained from SVM: " + str(round(np.mean([x[1] for x in scores]), 2)))
    print("Recall obtained from SVM: " + str(round(np.mean([x[2] for x in scores]), 2)))
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)

    return classifier_linear

if __name__ == '__main__':
    review = readCSV('Product_Review.csv')
    classifier_linear=classify_support_vector_machine(0.001,review)