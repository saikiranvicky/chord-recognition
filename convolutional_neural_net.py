import librosa
import pandas as pd
import numpy as np
import random
from keras.utils import to_categorical
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import h5py
import warnings
warnings.filterwarnings('ignore')


def cnn_recognition():

    # Load the data from chords.csv
    df = pd.read_csv('chords.csv')
    data = []
    for i in df.itertuples():
        # print(i[1])
        y, sr = librosa.core.load(i[1], duration=1.5)
        mfcc = librosa.feature.melspectrogram(y=y, sr=sr)
        # print(mfcc.shape)
        if mfcc.shape == (128, 65):
            data.append((mfcc, i[3]))
    print("number of audio samples : " + str(len(data)))

    # Shuffle the data randomly and load to training and testing sets
    random.shuffle(data)
    train = data[:1405]
    test = data[1405:]

    # Zip takes iterables and returns tuples
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    # Reshape the spectogram to (128,65)
    X_train = np.array([x.reshape((128, 65, 1)) for x in X_train])
    X_test = np.array([x.reshape((128, 65, 1)) for x in X_test])

    # One hot encoding to model class_id
    y_train = np.array(to_categorical(y_train, 10))
    y_test = np.array(to_categorical(y_test, 10))

    # Building Sequential Model
    model = Sequential()
    input_shape = (128, 65, 1)
    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

    # Train the Model
    hist = model.fit(x=X_train, y=y_train, epochs=40, batch_size=30, validation_data=(X_test, y_test))

    # Evaluation of the Model
    score = model.evaluate(x=X_test, y=y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    img = 'Test_Loss_and_Test_Accuracy'
    y_pos = np.arange(2)
    column = ['Test Loss', 'Test Accuracy']
    plt.bar(y_pos,score, label = 'Loss and Accuracy')
    plt.xticks(y_pos, column)
    plt.ylabel('Percentage')
    plt.legend()
    plt.savefig('images/{}'.format(img))
    plt.show()


    # Saving the accuracy and loss of the model in txt file
    with open('model_accuracy_and_loss.txt','w') as f:
        f.write('Test Loss : ' + str(score[0]) + '\n')
        f.write('Test Accuracy : ' + str(score[1]))

    train_loss = hist.history['loss']
    validation_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    validation_acc = hist.history['val_accuracy']
    num_epochs = range(1, 41)

    #Save the model loss to result_images
    name1 = 'model_loss'
    # Plotting Model Loss
    plt.figure(1, figsize=(8, 6))
    plt.plot(num_epochs, train_loss)
    plt.plot(num_epochs, validation_loss)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.grid(True)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig('images/{}'.format(name1))
    plt.show()

    # Saving the model accuracy to result_images
    name2 = 'model_accuracy'
    # Plotting Model Accuracy
    plt.figure(2, figsize=(8, 6))
    plt.plot(num_epochs, train_acc)
    plt.plot(num_epochs, validation_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.grid(True)
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.savefig('images/{}'.format(name2))
    plt.show()

    # Predicting the Model
    y_pred = model.predict_classes(X_test)
    label_id = np.argmax(y_test, axis=1)
    conf_matrix = confusion_matrix(label_id, y_pred, binary=False)
    print(conf_matrix)

    # Visualizing the performance of the Model
    name3 = 'confusion_matrix'
    plot_confusion_matrix(conf_mat=conf_matrix, class_names=chord_label)
    plt.title('Confusion Matrix')
    plt.savefig('images/{}'.format(name3))
    plt.show()

    # SAVE THE MODEL
    model.save('model.h5')


if __name__ == "__main__":
    chord_label = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']
    cnn_recognition()