import os
import audio_prep
import decoders

# Mapping from the model path to its decoder
datasets = {
    "Acoustic Datasets/ASVP-ESD-Update": decoders.decode_ASVP_ESD,
    "Acoustic Datasets/Emotion Speech Dataset": decoders.decode_ESD,
}

if __name__ == '__main__':
    for dataset in datasets:
        print(f"Now parsing dataset {os.path.basename(dataset)}...")
        audio_prep.prepare_audio(folder_path=dataset, decoder=datasets[dataset])
