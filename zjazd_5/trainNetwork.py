import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential


def trainNetwork(csvName, message):
    """
    Neural networks train
    :param csvName:
    """
    df = pd.read_csv(csvName)

    x = None
    y = None

    if csvName == 'data/banknotes.csv':
        df['is_authentic'] = [1 if Class == 1 else 0 for Class in df['Class']]
        df.drop('Class', axis=1, inplace=True)

        x = df.drop('is_authentic', axis=1)
        y = df['is_authentic']
    elif csvName == 'data/transfusion.csv':
        df['donated'] = [1 if donated_blood_in_March_2007 == 1 else 0 for donated_blood_in_March_2007 in
                         df['donated_blood_in_March_2007']]
        df.drop('donated_blood_in_March_2007', axis=1, inplace=True)

        x = df.drop('donated', axis=1)
        y = df['donated']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=len(df.columns) - 1))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

    model.fit(x_train, y_train, epochs=200, batch_size=32)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print(message, test_acc)
