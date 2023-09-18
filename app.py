import streamlit as st
import torchaudio
import torch
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa

st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_pitch(waveform, sr, pitch):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sr
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    
    # Convert waveform and pitch to NumPy arrays with detach()
    waveform_np = waveform[0].detach().numpy()
    pitch_np = pitch[0].detach().numpy()
    
    axis.plot(time_axis, waveform_np, linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch_np, linewidth=2, label="Pitch", color="green")
    st.pyplot(figure)

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
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
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
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)
    sd.wait()
    st.sidebar.text("Audio recorded successfully!")

    # Save the audio data as a WAV file
    filename = "audiofiles/download.wav"
    wav.write(filename, sample_rate, audio_data)
    st.sidebar.text(f"Audio saved as {filename}")

# Main content for audio analysis
if "audio_data" in locals():
    # Load the recorded audio file
    waveform, sample_rate = torchaudio.load(filename)

    # Display the waveform
    st.subheader("Audio Waveform")
    plot_waveform(waveform, sample_rate, title="Audio Waveform")

    # Resample (if needed)
    target_sample_rate = 16000  # Example target sample rate
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Calculate MFCCs
    n_mfcc = 10  # Number of MFCC coefficients
    n_fft = 20  # You can adjust this based on your audio data
    hop_length = 10  # You can adjust this based on your audio data

    # Calculate the maximum possible padding
    max_padding = n_fft - hop_length

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=target_sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': max_padding, 'hop_length': hop_length}
    )

    mfcc = mfcc_transform(waveform)

    # Display MFCCs
    st.subheader("MFCC")
    for channel in range(mfcc.shape[0]):
        plot_spectrogram(mfcc[channel], title="MFCC")

    # Calculate Pitch
    n_steps = 2  # Number of semitones to shift the pitch (adjust as needed)
    pitch_transform = torchaudio.transforms.PitchShift(sample_rate=target_sample_rate, n_steps=n_steps)
    pitched_waveform = pitch_transform(waveform)

    # Display Pitch-shifted waveform
    st.subheader("Pitch-shifted Audio")
    plt.figure(figsize=(10, 4))
    for channel in range(pitched_waveform.shape[0]):
        plt.plot(pitched_waveform[channel].detach().numpy())  # Use detach() here
    plt.title("Pitch-shifted Audio")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    st.pyplot()

    # Plot Pitch
    pitch_transform = torchaudio.transforms.Resample(orig_freq=target_sample_rate, new_freq=100)
    pitched_waveform_resampled = pitch_transform(pitched_waveform)
    pitch = torchaudio.transforms.MFCC(n_mfcc=1, melkwargs={'n_fft': n_fft, 'hop_length': hop_length})(pitched_waveform_resampled)
    plot_pitch(pitched_waveform_resampled, 100, pitch)

    # Spectrograms
    st.subheader("Spectrograms")

    # Spectrogram
    spectrogram_transform = torchaudio.transforms.Spectrogram(
    n_fft=n_fft,
    hop_length=hop_length
    )
    spectrogram = spectrogram_transform(waveform)

    # Display the spectrogram using matplotlib
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram[0].detach().numpy(), cmap='viridis', origin='lower', aspect='auto', extent=[0, duration, 0, target_sample_rate / 2])
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format="%+2.0f dB")
    st.pyplot(plt.gcf())  # Use st.pyplot to display the figure in Streamlit

    # Mel Spectrogram
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length
    )(waveform)
    
    mel_spectrogram = spectrogram_transform(waveform)
    # Display the spectrogram using matplotlib
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram[0].detach().numpy(), cmap='viridis', origin='lower', aspect='auto', extent=[0, duration, 0, target_sample_rate / 2])
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format="%+2.0f dB")
    st.pyplot(plt.gcf())  # Use st.pyplot to display the figure in Streamlit

    # Audio Player
    st.subheader("Audio Player")
    st.audio(filename)
    

    # Get other information
    num_channels = waveform.shape[0]
    num_frames = waveform.shape[1]
    duration = num_frames / target_sample_rate

    st.subheader("Audio Information")
    st.write(f"Number of Channels: {num_channels}")
    st.write(f"Number of Frames: {num_frames}")
    st.write(f"Duration (seconds): {duration:.2f}")
