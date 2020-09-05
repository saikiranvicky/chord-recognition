# chord-recognition
Chord Recognition using Template Based Matching and Convolutional Neural Network.

1)chroma_features.py 
In this py file, the audio is converted to chroma features using Short Time Fourier Transform, Constant Q Transform , Chroma Energy Normalized and Melspectogram spectral Features.

2)template_based_matching.py
In this py file, Chords are predicted by mapping the chroma features with the labeled chord templates. In TBM, I have used STFT to compute chroma features. 

4)audio_to_csv.py
In this py file, To train the neural networks, all the audio data is stored in the csv file with audio path, chord id, chord name.
The results are stored in chords.csv  

5)convolutional_neural_net.py
In this py file, The audio is converted to chroma features using melspectogram. Melspectograms and chord id are used to train the neural networks with melspectograms as input and chord id as output. I have plotted graphs of model loss, model accuracy and confusion matrix.

6)cnn_test_new_data.py
In this py file, the neural network that we trained is tested on completely new data. Results are stored in actual_chord_vs_predicted_chord.txt

7)chord_recognition_Cnn.py 
In this py file, a single audio file is passed to predict the chord label. 

All the plots and graphs are stored in the images folder.

DataSet To Train the Neural Network :
The dataset was downloaded from University of Leige website : https://people.montefiore.uliege.be/josmalskyj/research.php
The dataset contains 2000 audio files of 10 chords. Each chord has 200 audio files. 
After Precprocessing 1757 audio files of same frame size are used to train the CNN.  

chords.csv stores the audio path, chord name and chord id for jim2012Chords/Guitar_Only folder.

chords_test.csv stores the audio file and chrod id for jim2012Chords/Other_Instruments/Guitar. 

predicted_Chords_using_TBM.txt, predicted_chords_using_viterbi.txt,
model_accuracy_and_loss.txt, model_evalution_on_new_data.txt, actual_chord_vs_predicted_chord.txt are the results stored directly in the folder. 

chords are predicted in tbm, hmm using single audio file.

Matplotlib and Librosa are used to plot the graphs. 

There are some other audio files of chords in sample_chords and wav folder. You can try them to template_based_matching.py, chord_recognition_cnn.py as these py files are run on single audio file to get different results.
