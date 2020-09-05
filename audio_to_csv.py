import os
import csv


# Function to get path of each audio file
def create_metadata(data_path):
    chord_label = os.listdir(data_path)
    #print(chord_label)
    audio = list()
    for data in chord_label:
        audio_path = os.path.join(data_path, data)
        if os.path.isdir(audio_path):
            audio = audio + create_metadata(audio_path)
        else:
            audio.append(audio_path)
    return audio


# Write the audio path , chord_label and class_id to chords.csv file
def write_csv(audio_file,my_dict):
    print('\n Audio files to train the neural networks : ' + '\n')
    with open('chords.csv', 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        for data in audio_file:
            print(data)
            x = data.split('\\')
            for key in my_dict:
                if x[2] == key:
                    csv_writer.writerow([data, x[2], my_dict[key]])


def write_csv_test(test_audio, my_dict):
    print('\n Audio files to test neural network : ' +'\n')
    with open('chords_test.csv', 'w') as file:
        csv_write = csv.writer(file, delimiter=',')
        csv_write.writerow(['AUDIO FILE', 'CLASS_ID'])
        for audio in test_audio:
            print(audio)
            x = audio.split('\\')
            for k in my_dict:
                if x[3] == k:
                    csv_write.writerow([audio, my_dict[k]])


def main():
    data_path = 'jim2012Chords\Guitar_Only'
    data_test_path = 'jim2012Chords\Other_Instruments\Guitar'
    all_files = create_metadata(data_path)
    all_test_files = create_metadata(data_test_path)
    chords_dict = {
        "a": 0, "am": 1, "bm": 2, "c": 3, "d": 4, "dm": 5, "e": 6, "em": 7, "f": 8, "g": 9
    }
    write_csv(all_files, chords_dict)
    write_csv_test(all_test_files, chords_dict)


if __name__ == "__main__":
    main()
