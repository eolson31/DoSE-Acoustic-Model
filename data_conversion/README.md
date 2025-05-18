# Data Conversion

This code is responsible for converting a dataset into a valid format for the model. It will save its results into a folder named `train_data`. The code will not run as provided, minor changes need to be made.

## Usage

1. Download the audio file you wish to use for training.
    - For the DoSE App model, the [ASVP-ESD](https://www.kaggle.com/datasets/dejolilandry/asvpesdspeech-nonspeech-emotional-utterances) and [ESD](https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd) datasets were used.
2. A dictionary is defined within `main.py` named `datasets`. The left side of this dictionary holds paths to downloaded datasets, while the right side is their decoder. Remove the existing paths and add your path to the dataset(s) you are interested in preparing. You can add or remove lines from the dictionary. 
3. Each dataset needs its own decoder, that is, a function which tells the correct emotion (anger, happy, neutral, sad, or surprised) for a given audio clip in the dataset. The decoders are located in `decoders.py`. Here, you will see the decoders used for the ASVP-ESD and ESD datasets as examples along with an empty template at the top of the file. If you use a different dataset, you will need to write your own decoder for it.
4. Add your decoder(s) to the right side of the dictionary in `main.py`.
5. Assuming your path(s) and decoder(s) are correct, you can run the program using `python3 data_conversion/main.py`. This program takes a long time to run, as it converts the entire dataset into trainable Mel spectrograms. However, you should only need to run this once, as the results are cached as .npy files for the model and are saved to a folder named `train_data`.
