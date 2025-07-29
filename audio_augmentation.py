import torch_audiomentations as ta
import soundfile as sf
import torch
import os
import argparse
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

class AudioMask(nn.Module):
    def __init__(self, mask_ratio=0.025, num_mask=2, p=1):
        super(AudioMask, self).__init__()
        self.mask_ratio = mask_ratio
        self.num_mask = num_mask
        self.p = p
    
    def forward(self, wav: torch.Tensor):
        if random.random() > self.p:
            for _ in range(self.num_mask):
                batch_size, num_channels, num_frames = wav.shape
                mask_length = int(num_frames * self.mask_ratio)
                start = random.randint(0, num_frames - mask_length)
                end = start + mask_length
                
                # Create a mask
                wav[:, :, start:end] = 0
        return wav

class AudioAugmentation(nn.Module):
    def __init__(self, device='cpu'):
        super(AudioAugmentation, self).__init__()

        self.device = device

        self.id2transform = {
            0: 'ColouredNoise',
            1: 'BandPassFilter',
            2: 'BandStopFilter',
            3: 'Gain',
            4: 'LowPassFilter',
            5: 'Normalization',
            6: 'PitchShift',
            7: 'AudioMask'
        }
        self.num_transforms = len(self.id2transform)
        self.transform_ids_list = [i for i in range(self.num_transforms)]
        self.random_weights = [1 for _ in range(self.num_transforms)]

        self.transforms = self.get_dict_transforms()

    def forward(self, wav: np.ndarray, sample_rate=48000, mix_threshold=0.5):
        random_weights = self.random_weights.copy()
        
        transforms = []
        transform_ids = []
        transform_id = random.choices(self.transform_ids_list, weights=random_weights, k=1)[0]
        # transform_id = 6
        transform_ids.append(transform_id)
        transform_name = self.id2transform[transform_id]
        transform_names = [transform_name]

        random_weights[transform_id] = 0  # Prevent re-selection of the same transform
        
        more_transform = random.random()
        while more_transform > mix_threshold and sum(random_weights) > 0:
            transform_id = random.choices(self.transform_ids_list, weights=random_weights, k=1)[0]
            transform_ids.append(transform_id)
            transform_name = self.id2transform[transform_id]
            transform_names.append(transform_name)
            random_weights[transform_id] = 0  # Prevent re-selection of the same transform
            more_transform = random.random()
        
        audio_mask = None
        for transform_id in transform_ids:
            if transform_id == 6:
                transform = ta.PitchShift(p=1, sample_rate=sample_rate, min_transpose_semitones=-8, max_transpose_semitones=8, output_type='tensor')
                transforms.append(transform)
            elif transform_id == 7:
                audio_mask = AudioMask(mask_ratio=0.05, num_mask=2, p=1)
            else:
                transform = self.transforms[self.id2transform[transform_id]]
                transforms.append(transform)
        
        transform = ta.Compose(transforms, p=1, output_type='tensor')

        wav = wav.copy()
        wav = self.prepocess(wav).to(self.device)
        # print(wav.shape)
        wav = transform(wav, sample_rate=sample_rate).to(torch.float32)
        if audio_mask is not None:
            wav = audio_mask(wav)
        wav = wav.squeeze(0)  # Remove batch dimension
        wav = wav.cpu().numpy()
        name = '_'.join(transform_names)
        return wav, sample_rate, name

    def get_dict_transforms(self):
        transforms = {
            'ColouredNoise': ta.AddColoredNoise(p = 1, output_type='tensor'),
            # 'ImpulseResponse': ta.ApplyImpulseResponse(p = 0.5),
            # 'BackgroundNoise': ta.AddBackgroundNoise(p = 1),
            # 'PolarityInversion': ta.PolarityInversion(p = 1),
            'BandPassFilter': ta.BandPassFilter(p = 1, output_type='tensor'),
            'BandStopFilter': ta.BandStopFilter(p = 1, output_type='tensor'),
            'Gain': ta.Gain(min_gain_in_db=-8, max_gain_in_db=8, p = 1, output_type='tensor'),
            'LowPassFilter': ta.LowPassFilter(min_cutoff_freq=1000, max_cutoff_freq=2000, output_type='tensor'),
            # 'PitchShift': ta.PitchShift(p = 1, sample_rate= sample_rate, min_transpose_semitones=-8, max_transpose_semitones=8),
            'Normalization': ta.PeakNormalization(p = 1, output_type='tensor'),
        }
        return transforms
    
    def prepocess(self, wav: np.array):
        wav = torch.from_numpy(wav[None, :, :]).to(torch.float32)
        return wav

if __name__ == '__main__':
    file_path = './SPEAKER_00_1.14_36.3_6.wav'
    wav, sr = sf.read(file_path, always_2d=True)
    wav = wav.transpose(1, 0)
    # print(wav.shape, sr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio_augmentation = AudioAugmentation(device=device)
    augmented_wav, sample_rate, name = audio_augmentation(wav, sr)
    print(augmented_wav.shape, sample_rate, name)
