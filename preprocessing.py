import librosa
import numpy as np
from scipy.io import wavfile
import logging
from scipy import signal
import os

logger_module_name = 'preprocessing'
logger = logging.getLogger('step_plus.' + logger_module_name)


def mfcc(audio_filename):
    log_S = mel_spectrogram(audio_filename)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=40)
    mfcc = librosa.feature.delta(mfcc, order=2)
    return mfcc.T


def resample_audio_file(samples, sample_rate, new_sample_rate=16000):
    resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
    return resampled, new_sample_rate


def mel_spectrogram(audio_filename, resampled=True, max_len=10, normalize=False):
    sample_rate, samples = wavfile.read(audio_filename)

    if samples.shape[0] < sample_rate * max_len:
        reshaped_samples = np.zeros((sample_rate * max_len,))
        reshaped_samples[:samples.shape[0]] = samples
    else:
        reshaped_samples = samples[:(max_len * sample_rate)]

    if resampled:
        reshaped_samples, sample_rate = resample_audio_file(reshaped_samples, sample_rate)

    S = librosa.feature.melspectrogram(reshaped_samples, sr=sample_rate, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    # z-score normalization
    if normalize:
        mean = log_S.mean()
        std = log_S.std()
        if std != 0:
            log_S -= mean
            log_S /= std

    return log_S.T