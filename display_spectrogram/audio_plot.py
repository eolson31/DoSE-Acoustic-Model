import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

SAMPLE_RATE = 16000  # Standard sampling rate for speech
N_FFT = 1024  # FFT window size
HOP_LENGTH = 512  # Hop length for overlapping windows
N_MELS = 128  # Number of Mel filter banks
AUDIO_LENGTH = 3  # Target duration of audio clips in seconds


def load_audio(audio_path: str): # OLD FUNCTION, STILL USED FOR WAVE PLOT
    """Loads an audio file and returns the waveform and sample rate."""
    # Load audio file
    audio, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Pad or truncate audio to fixed length
    target_length = AUDIO_LENGTH * sample_rate
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    return audio, sample_rate

def plot_waveform(audio, sample_rate):
    """Plots the waveform of the audio."""
    plt.figure()
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def audio_to_mel_spectrogram(audio, sample_rate):
    """Converts audio to a Mel spectrogram and normalizes it."""
    # Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if np.min(log_mel_spectrogram) > 0:
        # Normalize range to [0, 1]
        log_mel_spectrogram = (log_mel_spectrogram - np.min(log_mel_spectrogram)) / (np.min(log_mel_spectrogram))

    return log_mel_spectrogram, sample_rate

def plot_mel_spectrogram(mel_spectrogram, sample_rate):
    """Plots the Mel spectrogram."""
    # Plot the log-Mel spectrogram
    plt.figure()
    librosa.display.specshow(mel_spectrogram, sr=sample_rate, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
    plt.colorbar(label="Normalized Log Power")
    plt.title("Mel Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Frequency")
    plt.show()
