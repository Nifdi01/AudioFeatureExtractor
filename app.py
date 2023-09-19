import torch
import streamlit as st
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
from IPython.display import Audio

st.set_option('deprecation.showPyplotGlobalUse', False)


def plot_pitch(waveform, sr, pitch):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sr
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

    # axis2.legend(loc=0)
    st.pyplot(figure)  # Display the Matplotlib figure in Streamlit


def plot_waveform(waveform, sr, title="Waveform"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    plt.figure()
    plt.plot(time_axis, waveform[0], linewidth=1)
    plt.grid(True)
    plt.title(title)
    st.pyplot()


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram),
                    origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    st.pyplot(fig)


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    st.pyplot(fig)


# Define the Streamlit app
st.title("Audio Feature Extractor")

# Sidebar for recording audio
st.sidebar.header("Audio Recorder")

duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, 5)
sample_rate = st.sidebar.selectbox("Sample Rate", [44100, 22050, 16000])
record_button = st.sidebar.button("Record Audio")


if record_button:
    # Record audio
    st.sidebar.text("Recording...")
    audio_data = sd.rec(int(duration * sample_rate),
                        samplerate=sample_rate, channels=2)
    sd.wait()
    st.sidebar.text("Audio recorded successfully!")

    # Save the audio data as a WAV file
    filename = "audiofiles/output.wav"
    wav.write(filename, sample_rate, audio_data)

# Main content for audio analysis
if "audio_data" in locals():
    # Load the recorded audio file
    waveform, sample_rate = torchaudio.load(filename)

    # Audio Player
    st.subheader("Audio Player")
    st.audio(filename)

    # Display the waveform
    st.subheader("Audio Waveform")
    st.write("This is the visual representation of the original audio input")
    plot_waveform(waveform, sample_rate, title="Audio Waveform")

    # Resample (if needed)
    target_sample_rate = 16000  # Example target sample rate
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Spectogram
    st.subheader("Spectrogram")
    st.write("A **spectrogram** is a visual representation of how the frequencies in an audio signal \
    change over time. It displays time on the x-axis, frequency on the y-axis, and color/intensity \
    to represent the strength of each frequency at different moments. It's essential for analyzing \
    audio content, such as identifying sounds or music patterns.")

    n_fft = 1024
    win_length = None
    hop_length = 512

    # Define transform
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )
    spectrogram = spectrogram_transform(waveform)

    plot_spectrogram(specgram=spectrogram[0], title='Spectogram')

    # Mel Filter Bank
    st.subheader("Mel Filter Bank")
    st.write("**Mel Filter Banks** are filters used to break down an audio signal into \
        different frequency bands, emphasizing lower frequencies similar to human \
        hearing perception. They're crucial for extracting relevant audio features, like MFCCs and LFCCs.")
    n_fft = 256
    n_mels = 64
    sample_rate = 6000

    mel_filters = F.melscale_fbanks(
        int(n_fft // 2 + 1),
        n_mels=n_mels,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        sample_rate=sample_rate,
        norm="slaney",
    )

    plot_fbank(mel_filters, "Mel Filter Bank")

    # MelSpectogram
    st.subheader("MelSpectogram")
    st.write("**MelSpectgrams** combines the principles of Mel Filter Banks and spectrograms. \
        Instead of showing the raw frequency content, it displays energy in Mel \
        frequency bands over time. This representation is more aligned with \
        human hearing and is widely used in speech and audio processing.")
    
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec = mel_spectrogram(waveform)

    plot_spectrogram(melspec[0], title="MelSpectrogram", ylabel="mel freq")

    # MFCC
    st.subheader("MFCC")
    st.write("**MFCCs (Mel-frequency cepstral coefficients)** are a compact representation \
        of audio that captures spectral and temporal features. They involve applying \
        Mel Filter Banks, taking the logarithm, and performing a cosine transform \
        to extract key audio characteristics for tasks like speech recognition.")
    
    n_fft = 2048
    win_length = None
    hop_length = 512
    n_mels = 256
    n_mfcc = 256

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "mel_scale": "htk",
        },
    )

    mfcc = mfcc_transform(waveform)

    plot_spectrogram(mfcc[0], title="MFCC")

    # LFCC
    st.subheader("LFCC")
    st.write("**LFCC (Linear frequency cepstral coefficients)** is similar to MFCCs \
        and represent audio features but use a linear scale rather than the Mel scale. \
        They are an alternative for audio analysis tasks.")
    
    n_fft = 2048
    win_length = None
    hop_length = 512
    n_lfcc = 256

    lfcc_transform = T.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
        },
    )

    lfcc = lfcc_transform(waveform)
    plot_spectrogram(lfcc[0], title="LFCC")

    # Pitch
    st.subheader("Pitch")
    st.write("**Pitch** refers to the perceived frequency of a sound. It's how we \
        interpret high or low musical notes or the tone of someone's voice. \
        Pitch is crucial in music and speech analysis, influencing aspects \
        like melody recognition and emotional tone detection.")
    
    pitch = F.detect_pitch_frequency(waveform, sample_rate)
    plot_pitch(waveform, sample_rate, pitch)

    # Get other information
    num_channels = waveform.shape[0]
    num_frames = waveform.shape[1]
    duration = num_frames / target_sample_rate

    st.subheader("Audio Information")
    st.write("**Channel Definitions**")
    st.write("Mono (1 Channel): _Mono audio contains a single audio channel, \
        which means there is only one audio source, and the sound is \
        perceived as coming from a single point._")
    st.write("Stereo (2 Channels): _Stereo audio consists of two channels, \
        typically labeled as left and right. This setup creates a sense of \
        direction and spatial separation, enhancing the perception of sound in two dimensions._")
    st.write("Multichannel (More than 2 Channels): _Multichannel audio can have more than \
        two channels.This is commonly used in home theater systems._")
    
    st.write(f"Number of Channels: {num_channels}")
    st.write(f"Number of Frames: {num_frames}")
    st.write(f"Duration (seconds): {duration:.2f}")
