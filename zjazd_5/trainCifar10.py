import tensorflow as tf
from keras import layers, models
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def trainCifar10():
    """
    Animals recognition neural network based on cifar10
    """
    cifar10 = tf.keras.datasets.cifar10

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('Cifar10 accuracy:', test_acc)

    # confusion matrix
    y_pred = model.predict(test_images)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(test_labels, axis=1)
    confusion_mtx = tf.math.confusion_matrix( y_pred_classes, y_true)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(12, 9))
    c = sns.heatmap(confusion_mtx, annot=True, fmt='g')
    c.set(xticklabels=classes, yticklabels=classes)
    plt.show()
