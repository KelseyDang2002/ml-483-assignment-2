import pandas as pd
import re
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

TEST_DATA_PERCENTAGE = 0.2 # 20% testing and 80% training

'''Main'''
def main():
    # download stopwords
    nltk.download('stopwords')

    # read csv file
    dataframe = pd.read_csv('emails.csv')

    # preprocess the text
    dataframe['cleaned_text'] = dataframe['text'].apply(preprocess)
    
    # convert text into numerical features using CountVectorizer
    cv = CountVectorizer()
    x = cv.fit_transform(dataframe['cleaned_text']).toarray()

    # target variable
    y = dataframe['spam']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TEST_DATA_PERCENTAGE, random_state=0)
    print(f"\n{(1 - TEST_DATA_PERCENTAGE) * 100}% Training {TEST_DATA_PERCENTAGE * 100}% Testing")

    # model training
    svm_model(X_train, X_test, y_train, y_test)
    naive_bayes_model(X_train, X_test, y_train, y_test)

    # model training with improvements
    svm_model_improved(X_train, X_test, y_train, y_test)
    naive_bayes_model_improved(X_train, X_test, y_train, y_test)

'''Preprocess email text'''
def preprocess(text):
    # initialize stemmer and stopwords
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # remove special characters
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)

    # convert to lowercase
    text = text.lower()

    # tokenize by splitting the text into words
    words = text.split()

    # remove stopwords and apply stemming
    words = [ps.stem(word) for word in words if word not in stop_words]

    # rejoin the words into a single string
    return ' '.join(words)

'''Function to plot ROC curve'''
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')

'''SVM model training + evaluation'''
def svm_model(X_train, X_test, y_train, y_test):
    # SVM model
    print("\nTraining SVM Model...")
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    # predictions for SVM
    y_pred_svm = svm_model.predict(X_test)

    # evaluate SVM model
    print("\nSVM Model Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm) * 100}%")
    print(f"Precision: {precision_score(y_test, y_pred_svm) * 100}%")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_svm)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred_svm)}")

    # generate ROC curve for SVM
    y_scores_svm = svm_model.decision_function(X_test)  # decision function for SVM
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_scores_svm)
    roc_auc_svm = roc_auc_score(y_test, y_scores_svm)

    # plot ROC curve for SVM
    plt.figure()
    plot_roc_curve(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.2f})")
    plt.legend(loc='lower right')
    plt.show()

'''Naive Bayes model training + evaluation'''
def naive_bayes_model(X_train, X_test, y_train, y_test):
    # Naive Bayes (multinomial) model
    print("\nTraining Naive Bayes Model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # predictions for Naive Bayes
    y_pred_nb = nb_model.predict(X_test)

    # evaluate Naive Bayes model
    print("\nNaive Bayes Model Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_nb) * 100}%")
    print(f"Precision: {precision_score(y_test, y_pred_nb) * 100}%")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_nb)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred_nb)}")

    # generate ROC curve for Naive Bayes
    y_proba_nb = nb_model.predict_proba(X_test)[:, 1]  # probability estimates for class 1 (spam)
    fpr_nb, tpr_nb, _ = roc_curve(y_test, y_proba_nb)
    roc_auc_nb = roc_auc_score(y_test, y_proba_nb)

    # plot ROC curve for Naive Bayes
    plt.figure()
    plot_roc_curve(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC = {roc_auc_nb:.2f})")
    plt.legend(loc='lower right')
    plt.show()

'''
Improve classification performance
Method 1: Change kernel function for SVM
Method 2: Tune alpha parameter for Naive Bayes
'''

'''SVM model with different kernel'''
def svm_model_improved(X_train, X_test, y_train, y_test):
    # SVM model
    print("\nTraining SVM Model with rbf kernel...")
    svm_model = svm.SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)

    # predictions for SVM
    y_pred_svm = svm_model.predict(X_test)

    # evaluate SVM model
    print("\nSVM Model with rbf kernel Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm) * 100}%")
    print(f"Precision: {precision_score(y_test, y_pred_svm) * 100}%")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_svm)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred_svm)}")

'''Naive Bayes model with alpha tuning'''
def naive_bayes_model_improved(X_train, X_test, y_train, y_test):
    # Naive Bayes (multinomial) model
    print("\nTraining Naive Bayes Model with alpha tuning...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)

    # predictions for Naive Bayes
    y_pred_nb = nb_model.predict(X_test)

    # evaluate Naive Bayes model
    print("\nNaive Bayes Model with alpha tuning Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_nb) * 100}%")
    print(f"Precision: {precision_score(y_test, y_pred_nb) * 100}%")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_nb)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred_nb)}")

'''Call main'''
if __name__ == "__main__":
    main()
