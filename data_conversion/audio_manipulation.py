import librosa
import numpy as np

SAMPLE_RATE = 16000 # Standard sampling rate for speech
N_FFT = 1024        # FFT window size
HOP_LENGTH = 512    # Hop length for overlapping windows
N_MELS = 128        # Number of Mel filter banks
AUDIO_LENGTH = 3    # Target duration of audio clips in seconds
MIN_LENGTH = 1      # Minimum length of an audio clip when splicing
STEP = 1            # Start offset, in seconds, on each iteration until AUDIO_LENGTH is reached

def splice_audio(audio_path: str):
    '''Split audio into chunks of 3 seconds'''
    audio, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)

    target_length = AUDIO_LENGTH * sample_rate
    minimum_length = MIN_LENGTH * sample_rate
    audio_segments = []

    for x in range(int(AUDIO_LENGTH / STEP)):
        spliced_audio = audio

        while len(spliced_audio) > target_length:
            audio_segments.append(spliced_audio[:target_length])
            spliced_audio = spliced_audio[target_length:]

        if len(spliced_audio) < target_length and len(spliced_audio) > minimum_length:
            spliced_audio = np.pad(spliced_audio, (0, target_length - len(spliced_audio)))
            audio_segments.append(spliced_audio)
        
        audio = audio[int(STEP * sample_rate):]
    
    return audio_segments, sample_rate


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

def bulk_audio_to_mel_spectrogram(audio, sample_rate):
    spectrograms = []
    for clip in audio:
        spectrogram, _ = audio_to_mel_spectrogram(clip, sample_rate)
        spectrograms.append(spectrogram)
    return spectrograms

def extract_spectrograms(audio_path: str):
    '''Used by model'''
    audio, sample_rate = splice_audio(audio_path)
    mel_spectrogram = bulk_audio_to_mel_spectrogram(audio, sample_rate)
    return mel_spectrogram

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

def bulk_audio_to_mel_spectrogram(audio, sample_rate):
    spectrograms = []
    for clip in audio:
        spectrogram, _ = audio_to_mel_spectrogram(clip, sample_rate)
        spectrograms.append(spectrogram)
    return spectrograms

def extract_spectrograms(audio_path: str):
    '''Used by model to get spectrograms from an audio clip'''
    audio, sample_rate = splice_audio(audio_path)
    mel_spectrograms = bulk_audio_to_mel_spectrogram(audio, sample_rate)
    return mel_spectrograms
