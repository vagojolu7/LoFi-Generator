import librosa
import soundfile as sf
import numpy as np
from scipy import signal
from scipy.io import wavfile
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from random import randint

# Load the audio file
audio_file_path = "./LoFiScript/the-beat-of-nature.mp3"
y, sr = librosa.load(audio_file_path, sr=44100)

# Add distortion
y = np.tanh(y)

# Reduce the frequency range
y = signal.decimate(y, q=2)

# Apply filtering effects
fc = 3000  # cutoff frequency of the filter
b, a = signal.butter(4, fc / (sr / 2), 'lowpass')
y = signal.filtfilt(b, a, y)

# Add vinyl noise
vinyl_noise_file_path = "./LoFiScript/Mixdown_Half_2.wav"
vinyl_noise, _ = librosa.load(vinyl_noise_file_path, sr=sr, mono=True)
new_vinyl_noise = vinyl_noise
random_np = np.random.uniform(low=-1.0, high=1.0, size=len(y))

while random_np.size != new_vinyl_noise.size:
    new_vinyl_noise = np.insert(new_vinyl_noise, new_vinyl_noise.size, [0.0])

vinyl_noise = random_np * new_vinyl_noise * 0.05
y += vinyl_noise

# Apply EQ
freqs = [500, 1000, 2000, 4000, 8000]
gains = [-3, -2, 2, 3, 1]
Qs = [0.5, 1.0, 1.5, 2.0, 2.5]

for f, Q in zip(freqs, Qs):
    b, a = signal.iirpeak(f, Q, fs=sr)
    y = signal.filtfilt(b, a, y)
    y *= 10

# Write the processed audio to a new file
output_file_path = "./LoFiScript/Jazzy_LOFI_2.wav"
sf.write(output_file_path, y, sr)
