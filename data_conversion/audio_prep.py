import numpy as np
import os
import audio_manipulation

emotion_to_number = {
        'Anger': 0,
        'Happy': 1,
        'Neutral': 2,
        'Sad': 3,
        'Surprise': 4,
}

def prepare_audio(folder_path, decoder):
    complete_count = 0
    audio, label = [], []

    basename = os.path.basename(folder_path)
    basepath = os.path.join("train_data", basename)
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    data_path = os.path.join(basepath, "data_cache.npy")
    label_path = os.path.join(basepath, "label_cache.npy")
    # Load audio and cache it
    for subdir, dirs, files in os.walk(folder_path):
        # Logging
        print(f"Currently in folder '{os.path.basename(subdir)}' ({len(dirs)} subfolders, {len(files)} files)")

        for file in files:
            if not file.endswith('wav'):
                print(f"WARNING: Non wav file found: '{file}' ({subdir}). Skipping...")
            else:
                audio_path = os.path.join(subdir, file)
                audio_emotion = decoder(audio_path, file)
                if audio_emotion: # if the emotion is not None
                    if not audio_emotion in emotion_to_number: 
                        raise TypeError(f"Your decoder returned an invalid emotion type '{audio_emotion}'. It must return one of the following {set(emotion_to_number.keys())} or None to skip that audio file")
                    spectrograms = audio_manipulation.extract_spectrograms(audio_path)
                    for spectrogram in spectrograms:
                        audio.append(spectrogram)
                        label.append(emotion_to_number.get(audio_emotion))
                    # Logging
                    complete_count += 1
                    if complete_count % 250 == 0:
                        print(f"Successfully parsed {complete_count} audio files so far")

    data = np.array(audio, dtype=np.float32)
    labels = np.array(label, dtype=np.float32)
    print("---------------RESULTS---------------")
    print(f"Successfully parsed a total of {complete_count} audio files")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    # Cache
    np.save(data_path, data)
    np.save(label_path, labels)
    print(f"Successfully saved data to '{basepath}'")
