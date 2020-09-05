import librosa
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from cnn_test_new_data import get_load_model


# Chord Recognition for single wav file
def test_on_single_audio_file(new_model, audio):
    y, sr = librosa.core.load(audio, duration=1.5)
    mfcc = librosa.feature.melspectrogram(y=y, sr=sr)
    #print(mfcc.shape)

    new_mfcc = np.array(mfcc.reshape(1, 128, 65, 1))
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Predicting the class_id of spectogram
    pred = new_model.predict_classes(new_mfcc)
    class_id = (pred[0])
    print('\n Predicted chord is : ' + chord_label[class_id] + '\n')

    # Save the image
    name = 'chord_recognition_using_cnn'

    # Plotting bar graph for recognized chord
    y_val = np.arange(len(chord_label))
    x_val = np.array(to_categorical(pred[0], 10))
    plt.bar(y_val, x_val, align='center')
    plt.xticks(y_val, chord_label)
    plt.title('Chord Recognition')
    plt.savefig('images/{}'.format(name))
    plt.show()


if __name__ == "__main__":
    chord_label = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']
    new_model = get_load_model()
    data_path = 'wav\\Grand Piano - Fazioli - major C middle.wav'
    test_on_single_audio_file(new_model, data_path)
