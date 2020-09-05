import warnings
import pandas as pd
from keras.models import load_model
import numpy as np
from keras.utils import to_categorical
import librosa
warnings.filterwarnings('ignore')


def get_chord_label(val):
    return chord_label[val]


def get_load_model():

    # #Load the Model
    new_model = load_model('model.h5')

    # #Compile the model
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return new_model


def test_new_data(new_model):

    # #New Test data
    data = pd.read_csv('chords_test.csv', skiprows=1)
    data_set = []
    for row in data.itertuples():
        y, sr = librosa.core.load(row[1], duration=1.5)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        # print(mel_spec.shape)
        if mel_spec.shape == (128, 65):
            data_set.append((mel_spec, row[2]))

    print('number of testing samples : ' + str(len(data_set)) + '\n')

    # Split the Data into tuple of X and Y
    X_test, Y_test = zip(*data_set)

    # Reshape X to (128,65) as we fit the model with (128,65)
    X_test = np.array([x.reshape((128, 65, 1)) for x in X_test])
    y_test = np.array(to_categorical(Y_test, 10))

    # Evaluation
    score = new_model.evaluate(x=X_test, y=y_test)
    print('MODEL LOSS : ' + str(score[0]))
    print('MODEL ACCURACY : ' + str(score[1]))

    with open('model_evalution_on_new_data.txt', 'w') as f:
        f.write('MODEL LOSS ON NEW DATA : ' + str(score[0]) + '\n')
        f.write('MODEL ACCURACY ON NEW DATA : ' + str(score[1]))


    # Prediction of model
    # Output class_id for every spectogram
    predictions = new_model.predict_classes(X_test)

    # Predicting Chord and storing it into csv file
    with open('actual_chord_vs_predicted_chord.txt', 'w') as f:

        for label in range(len(predictions)):

            f.write('Actual Chord is : ' + str(get_chord_label(Y_test[label])) + '\n')
            f.write('Predicted Chord is : ' + str(get_chord_label(predictions[label])) + '\n')
            print('Actual Chord: ' + get_chord_label(Y_test[label]))
            print('Predicted Chord: ' + get_chord_label(predictions[label]) + '\n')


if __name__ == "__main__":
    chord_label = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']
    new_model = get_load_model()
    test_new_data(new_model)
