# Authors: Marcin Żmuda-Trzebiatowski and Jakub Cirocki
# Example: https://github.com/s20501/NAI/blob/main/zjazd_4/example.PNG
#
# The program calculate precision of two different classification models

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics



def train(csvName):
    '''
    Download data from file. Train two models, one using SVC and other using Decision Tree.

    :param csvName: name of file containing data
    :return: Precision of each of classification models
    '''
    #Load data
    df = pd.read_csv(csvName)

    #Split data into parameters and result
    data = df.values
    X = data[:, :4]
    Y = data[:, 4]

    #Split data for training and testing models
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    #Train SVC
    svn = SVC(gamma=100)
    svn.fit(X_train, y_train)

    #Train Decision Tree
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    predictions = svn.predict(X_test)

    print(csvName)

    print("Decision tree accuracy score: ", metrics.accuracy_score(y_test, y_pred))

    print("SVC accuracy score: ", metrics.accuracy_score(y_test, predictions))
    print()

if __name__ == '__main__':
    train("banknotes.csv")
    train("transfusion.csv")



