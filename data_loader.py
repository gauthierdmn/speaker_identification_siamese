import torch.utils.data as data
from utils import list_files
from preprocessing import mel_spectrogram
import torch
import numpy as np
import random
import os


class SiameseDataset(data.Dataset):
    """Custom Dataset of Part A data compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, labels):
        """
        Set the path for audio data, together wth labels.
        """
        self.root = root
        self.labels = labels

    def __getitem__(self, index):
        """Returns one data pair (MFCC and label)."""
        list_audio_files = list_files(self.root)
        audio1 = list_audio_files[index]
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class is found
                audio2 = random.choice(list_audio_files)
                if self.labels[os.path.split(audio1)[1]] == self.labels[os.path.split(audio2)[1]]:
                    break
        else:
            audio2 = random.choice(list_audio_files)

        spect1, spect2 = torch.from_numpy(mel_spectrogram(os.path.join(audio1))).float(), torch.from_numpy(mel_spectrogram(os.path.join(audio2))).float()
        label1, label2 = self.labels[os.path.split(audio1)[1]], self.labels[os.path.split(audio2)[1]]
        return spect1, spect2, torch.from_numpy(np.array([int(label1!=label2)],dtype=np.float32))

    def __len__(self):
        return len(list_files(os.path.join(self.root)))
