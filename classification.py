import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

TEST_DATA_PERCENTAGE = 0.2

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

    # model training
    svm_model(X_train, X_test, y_train, y_test)
    naive_bayes_model(X_train, X_test, y_train, y_test)
    # best_svm_model = svm_with_grid_search(X_train, X_test, y_train, y_test)
    # best_svm_model = svm_with_grid_search(X_train, X_test, y_train, y_test)

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

# def svm_with_grid_search(X_train, X_test, y_train, y_test):
#     # Set the parameter grid for SVM
#     param_grid_svm = {
#         'C': [0.1, 1, 10, 100],  # Regularization parameter
#         'kernel': ['linear', 'rbf'],  # Kernel type
#         'gamma': ['scale', 'auto']  # Kernel coefficient
#     }
    
#     # Initialize the SVM model
#     svm_model = svm.SVC()

#     # Initialize GridSearchCV with 5-fold cross-validation
#     grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='accuracy', verbose=1)

#     # Fit the model with the training data
#     grid_search_svm.fit(X_train, y_train)

#     # Best parameters from the grid search
#     print(f"Best parameters for SVM: {grid_search_svm.best_params_}")
    
#     # Best model from grid search
#     best_svm_model = grid_search_svm.best_estimator_
    
#     # Make predictions
#     y_pred_svm = best_svm_model.predict(X_test)

#     # Evaluate the SVM Model
#     print("\nSVM Model Results with Grid Search:")
#     print("Accuracy:", accuracy_score(y_test, y_pred_svm))
#     print("Precision:", precision_score(y_test, y_pred_svm))
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
#     print("Classification Report:\n", classification_report(y_test, y_pred_svm))

#     # Return the best model to use for further evaluation or plotting
#     return best_svm_model

# def naive_bayes_with_grid_search(X_train, X_test, y_train, y_test):
#     # Set the parameter grid for Naive Bayes (only alpha for smoothing)
#     param_grid_nb = {
#         'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
#     }
    
#     # Initialize the Naive Bayes model
#     nb_model = MultinomialNB()

#     # Initialize GridSearchCV with 5-fold cross-validation
#     grid_search_nb = GridSearchCV(nb_model, param_grid_nb, cv=5, scoring='accuracy', verbose=1)

#     # Fit the model with the training data
#     grid_search_nb.fit(X_train, y_train)

#     # Best parameters from the grid search
#     print(f"Best parameters for Naive Bayes: {grid_search_nb.best_params_}")
    
#     # Best model from grid search
#     best_nb_model = grid_search_nb.best_estimator_
    
#     # Make predictions
#     y_pred_nb = best_nb_model.predict(X_test)

#     # Evaluate the Naive Bayes Model
#     print("\nNaive Bayes Model Results with Grid Search:")
#     print("Accuracy:", accuracy_score(y_test, y_pred_nb))
#     print("Precision:", precision_score(y_test, y_pred_nb))
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
#     print("Classification Report:\n", classification_report(y_test, y_pred_nb))

#     # Return the best model to use for further evaluation or plotting
#     return best_nb_model

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
    print("\nSVM Model Results:")
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

'''Call main'''
if __name__ == "__main__":
    main()
