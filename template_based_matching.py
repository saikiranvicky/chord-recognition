import librosa
import numpy as np
import matplotlib.pyplot as plt


# Load the audio file and compute time series using Librosa
def audio_series(audio_data):
    y, sr = librosa.core.load(audio_data,duration=3)
    return y,sr


# Defining Template
def template():
    maj_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    min_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    template_array = np.zeros((25, 12), dtype=float)
    for c in range(12):
        template_array[c, :] = np.roll(maj_template, c)
        template_array[c + 12, :] = np.roll(min_template, c)

    template_dict = {}
    my_list = template_array.tolist()

    for key in labels:
        for value in my_list:
            template_dict[key] = value
            my_list.remove(value)
            break

    return labels, template_array, template_dict


# Computing chroma vector using short time fourier transfrom
def calculate_chroma(samples, sr, wz, hs):
    chroma_feature = librosa.feature.chroma_stft(y=samples, sr=sr, n_fft=wz, hop_length=hs,
                                          tuning=librosa.core.estimate_tuning(y=samples, sr=sr))
    #print(chromas.shape)
    return chroma_feature


# Computing Cosine Similarity
def cosine_similarity(u, v):
    value = np.inner(u, v)/(np.linalg.norm(u) * np.linalg.norm(v))
    return value


# Predicting the chord by mapping Template with Chroma Vecotr
def cosine(chroma):
    label_list = list()
    chord_list = list()
    for i in range(chroma.shape[1]):
        for k, v in template_dict.items():
            temp = cosine_similarity(chroma[:, i], v)
            chord_list.append((k, temp))
        max_chord_list = max(chord_list, key=lambda x: x[1])[0]
        label_list.append(max_chord_list)
    with open('predicted_chords_using_TBM.txt', 'w') as f:
        f.write(str(label_list))
    print(label_list)
    return label_list


def plot_tbm(pcp, chord_predicted):
    my_dict = {}
    for i, chord in enumerate(label):
        my_dict[chord] = i

    my_list = []
    for x in chord_predicted:
        my_list.append(my_dict[x])
    my_list = np.array(my_list)

    # Saving the image in results_image folder
    name = 'chord_recognition_using_tbm'
    plt.figure()
    y_val = librosa.times_like(my_list)
    plt.scatter(y_val, my_list + 0.5, color='green', alpha=0.5, marker='+', label='Estimated Chord')
    plt.yticks(0.5 + np.unique(my_list), [labels[i] for i in np.unique(my_list)], va='center')
    plt.legend()
    plt.title('Template Based Matching')
    #plt.colorbar()
    plt.tight_layout()
    plt.savefig('images/{}'.format(name))
    plt.show()


if __name__ == "__main__":
    labels = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj',
              'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
              'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min',
              'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
              ]
    y, sample_rate = audio_series("wav\\d1.wav")
    label, weights, template_dict = template()
    chroma_vector = calculate_chroma(y, sample_rate, 4096, 1024)
    predicted_chords = cosine(chroma_vector)
    plot_tbm(chroma_vector,predicted_chords)