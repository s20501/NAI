import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score


def trainNetwork(csvName):
    """
    Neural networks train
    :param csvName:
    """
    df = pd.read_csv(csvName)

    df['is_authentic'] = [1 if Class == 1 else 0 for Class in df['Class']]
    df.drop('Class', axis=1, inplace=True)

    x = df.drop('is_authentic', axis=1)
    y = df['is_authentic']



    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    model = Sequential()
    model.add(Dense(units=32,activation='relu',input_dim=4))
    model.add(Dense(units=64,activation='relu'))
    model.add(Dense(units=64,activation='relu'))
    model.add(Dense(units=1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

    model.fit(x_train, y_train, epochs=200, batch_size=32)
    y_hat = model.predict(x_test)
    y_hat = [0 if val < 0.5 else 1 for val in y_hat]


    print(csvName)
    print("Neural networks accuracy score: ",  accuracy_score(y_test, y_hat))











