import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

df = pd.read_csv('data.csv', names=['Variance', 'Skewness', 'Curtosis', 'Entropy','Class'])


data = df.values
X = data[:,:4]
Y = data[:,4]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

svn = SVC(kernel='rbf', C=1, gamma=100)
svn.fit(X_train, y_train)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

predictions = svn.predict(X_test)
confusion_matrix(y_test,predictions)

print(metrics.accuracy_score(y_test, y_pred))

print(metrics.accuracy_score(y_test, predictions))


