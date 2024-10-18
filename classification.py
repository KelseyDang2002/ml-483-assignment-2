import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, GaussianNB

TEST_DATA_PERCENTAGE = 0.2

'''Main'''
def main():
    # read csv file
    dataframe = pd.read_csv("emails.csv")
    print(dataframe.head())

    # separate spam labels from text
    x = dataframe['text']
    y = dataframe['spam']
    print(f"\nText:\n{x}")
    print(f"\nSpam Labels:\n{y}")

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TEST_DATA_PERCENTAGE, random_state=0)
    
    # extract training features
    cv = CountVectorizer()
    features = cv.fit_transform(X_train)

    model = svm.SVC()
    model.fit(features, y_train)

    # extract testing features
    features_test = cv.fit_transform(X_test)

    print(f"Accuracy: {model.score(features_test, y_test)}")

'''Call main'''
if __name__ == "__main__":
    main()
