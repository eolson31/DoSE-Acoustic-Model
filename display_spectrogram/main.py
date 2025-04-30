import audio_plot as audio_plot

if __name__ == '__main__':
    audio_path = 'display/audio.wav'
    audio, sample_rate = audio_plot.load_audio(audio_path)
    # audio_plot.plot_waveform(audio, sample_rate)

    mel_spectrogram, sample_rate = audio_plot.audio_to_mel_spectrogram(audio, sample_rate)
    audio_plot.plot_mel_spectrogram(mel_spectrogram, sample_rate)
