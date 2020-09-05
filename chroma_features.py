import librosa
import numpy as np
from librosa import display
import matplotlib.pyplot as plt


def plot_chroma_features(audio_file):

    # LOAD THE AUDIO DATA
    # LIBROSA IS A PYYTHON LIBRARY WHICH IS USED TO LOAD AUDIO DATA
    y, sr = librosa.load(audio_file)
    duration = len(y) / sr
    print('duration of the audio file : ' + str(duration))

    # Visualizing Audio
    name1 = 'Visualization_of_Audio'
    plt.figure()
    librosa.display.waveplot(y=y, sr=sr)
    plt.xlabel("Time in seconds")
    plt.ylabel("Amplitude")
    plt.savefig('images/{}'.format(name1))
    plt.show()

    # Visualization of STFT, CQT, CENS
    name2 = 'Chroma_vectors_using_STFT_CQT_CENS'
    chroma_stf = librosa.feature.chroma_stft(y=y, sr=sr)
    print(chroma_stf.shape)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    print(chroma_cq.shape)
    chroma_cen = librosa.feature.chroma_cens(y=y, sr=sr)
    print(chroma_cen.shape)
    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.specshow(chroma_stf, y_axis='chroma')
    plt.title('chroma_stf')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
    plt.title('chroma_cqt')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(chroma_cen, y_axis='chroma', x_axis='time')
    plt.title('chroma_cen')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('images/{}'.format(name2))
    plt.show()

    # Compute Chroma Vector using Melspectogram
    name3 = 'Chroma_Vector_using_Melspectogram'
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.figure()
    librosa.display.specshow(mel, y_axis='mel', x_axis='time')
    plt.title('melspectogram')
    plt.tight_layout()
    plt.savefig('images/{}'.format(name3))
    plt.show()

    # Computing Log power spectogram of melspectogram
    name4 = 'log_power_spectogram'
    S_DB = librosa.power_to_db(mel, ref=np.max)
    print(S_DB.shape)
    plt.figure()
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel');
    plt.tight_layout()
    plt.savefig('images/{}'.format(name4))
    plt.show()


if __name__ == '__main__':
    data_path = 'wav/c1.wav'
    plot_chroma_features(data_path)