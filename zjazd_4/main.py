import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics



def train(csvName):
    df = pd.read_csv(csvName)


    data = df.values
    X = data[:, :4]
    Y = data[:, 4]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    svn = SVC(gamma=100)
    svn.fit(X_train, y_train)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    predictions = svn.predict(X_test)
    confusion_matrix(y_test, predictions)

    print(csvName)

    print("Decision tree accuracy score: ", metrics.accuracy_score(y_test, y_pred))

    print("SVC accuracy score: ", metrics.accuracy_score(y_test, predictions))
    print()




if __name__ == '__main__':
    train("banknotes.csv")
    train("transfusion.csv")



