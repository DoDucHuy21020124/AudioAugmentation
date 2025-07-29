import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
import librosa
import random
import torch.nn.functional as F

class SpectrogramAugmentation(nn.Module):
    def __init__(
            self, 
            num_freq=1024, 
            hop_length=256, 
            win_length=1024,
            num_freq_mask=2,
            freq_percentage=0.025,
            num_time_mask=2,
            time_percentage=0.025,
            num_mels=80
        ):
        super(SpectrogramAugmentation, self).__init__()
        # Define spectrogram augmentation transforms here if needed
        self.num_freq = num_freq
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_freq_mask = num_freq_mask
        self.freq_percentage = freq_percentage
        self.num_time_mask = num_time_mask
        self.time_percentage = time_percentage
        self.num_mels = num_mels

        self.id2transform = {
            0: 'FreqMask',
            1: 'TimeMask',
            2: 'FreqResize',
            3: 'TimeResize',
        }

        self.num_transforms = len(self.id2transform)
        self.random_weights = [1 for _ in range(self.num_transforms)]
        self.transform_ids_list = [i for i in range(self.num_transforms)]

    def forward(
            self, 
            wav, 
            sample_rate=48000,
            mix_threshold=0.5
        ):
        # Apply frequency augmentations to the wav tensor
        # This is a placeholder; actual implementation will depend on the specific augmentations needed
        random_weights = self.random_weights.copy()
        transform_ids = []
        transform_id = random.choices(self.transform_ids_list, weights=random_weights, k=1)[0]
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

        wav = wav.copy()
        spec_mag, phase = self.get_spectrogram_phase(wav)
        for transform_id in transform_ids:
            if transform_id == 0:
                spec_mag = self.freq_spec_mask(spec_mag, num_freq_mask=self.num_freq_mask, freq_percentage=self.freq_percentage)
            elif transform_id == 1:
                spec_mag = self.time_spec_mask(spec_mag, num_time_mask=self.num_time_mask, time_percentage=self.time_percentage)
            elif transform_id == 2:
                spec_mag = self.freq_spec_resize(spec_mag, sample_rate=sample_rate, ratio=random.uniform(0.9, 1.1))
            elif transform_id == 3:
                spec_mag = self.time_spec_resize(spec_mag, sample_rate=sample_rate, ratio=random.uniform(0.9, 1.1))
        name = '_'.join(transform_names)
        wav = librosa.istft(spec_mag * phase, hop_length=self.hop_length, win_length=self.win_length)
        wav = wav.astype(np.float32)

        return wav, sample_rate, name
    
    def get_spectrogram_phase(self, wav):

        wav = wav.copy()
        # wav = wav.transpose(1, 0)  # Ensure the shape is (channels, samples)
        spec =  librosa.stft(wav, n_fft=self.num_freq, hop_length=self.hop_length, win_length=self.win_length)
        spec_mag, phase = librosa.magphase(spec)
        return spec_mag, phase
    
    def freq_spec_mask(self, spec: np.ndarray, num_freq_mask, freq_percentage):
        spec = spec.copy()
        for i in range(num_freq_mask):
            # get the number of frames and the number of frequencies.
            all_channels_num, all_freqs_num, all_frames_num = spec.shape
            # defines the amount of masking given a percentage.
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            # defines which frequency will be masked.
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            # rounds to an integer.
            f0 = int(f0)
            # masks the frequency by assigning zero.
            spec[:, f0:f0 + num_freqs_to_mask, :] = 0
        return spec

    def time_spec_mask(self, spec: np.ndarray, num_time_mask, time_percentage):
        spec = spec.copy()
        for i in range(num_time_mask):
            # get the number of frames and the number of frequencies.
            all_channels_num, all_freqs_num, all_frames_num = spec.shape
            # defines the amount of masking given a percentage.
            num_frames_to_mask = int(time_percentage * all_frames_num)
            # defines which instant of time will be masked.
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            # rounds to an integer.
            t0 = int(t0)
            # masks the instant of time by assigning zero.
            spec[:, :, t0:t0 + num_frames_to_mask] = 0

        return spec

    def freq_spec_resize(self, spec: np.ndarray, sample_rate: int, ratio: float):
        mel_spec = librosa.feature.melspectrogram(
            S=spec,
            sr=sample_rate,
            n_fft=self.num_freq,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.num_mels
        )
        num_channels, num_mels, num_frames = mel_spec.shape
        new_num_mels = int(num_mels * ratio)        

        mel_aug = torch.from_numpy(mel_spec).float()

        # Resize vertically using bilinear interpolation
        mel_resized = F.interpolate(mel_aug.unsqueeze(0), size=(new_num_mels, num_frames), mode='bilinear', align_corners=False).squeeze(0)
        if ratio < 1:
            pad_size = num_mels - new_num_mels
            top_pad = mel_resized[:, :1, :]  # Take highest freq bin
            noise = torch.randn_like(top_pad) * 0.01
            top_pad = top_pad + noise
            pad = top_pad.repeat(1, pad_size, 1)
            mel_aug = torch.cat([pad, mel_resized], dim=1)
        else:
            mel_aug = mel_resized[:, :new_num_mels, :]
        mel_aug = mel_aug.numpy()
        spec_aug = librosa.feature.inverse.mel_to_stft(
            mel_aug,
            sr=sample_rate,
            n_fft=self.num_freq,
        )

        return spec_aug

    def time_spec_resize(self, spec: np.ndarray, sample_rate: int, ratio: float):
        mel_spec = librosa.feature.melspectrogram(
            S=spec,
            sr=sample_rate,
            n_fft=self.num_freq,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.num_mels
        )
        num_channels, num_mels, num_frames = mel_spec.shape
        new_num_frames = int(num_frames * ratio)

        mel_aug = torch.from_numpy(mel_spec).float()

        # Resize vertically using bilinear interpolation
        mel_resized = F.interpolate(mel_aug.unsqueeze(0), size=(num_mels, new_num_frames), mode='bilinear', align_corners=False).squeeze(0)
        if ratio < 1:
            pad_size = num_frames - new_num_frames
            top_pad = mel_resized[:, :, -1:]  # Take highest time bin
            noise = torch.randn_like(top_pad) * 0.01
            top_pad = top_pad + noise
            pad = top_pad.repeat(1, 1, pad_size)
            mel_aug = torch.cat([mel_resized, pad], dim=2)
        else:
            mel_aug = mel_resized[:, :, :num_frames]
        mel_aug = mel_aug.numpy()
        spec_aug = librosa.feature.inverse.mel_to_stft(
            mel_aug,
            sr=sample_rate,
            n_fft=self.num_freq,
        )

        return spec_aug



if __name__ == '__main__':
    file_path = './SPEAKER_00_1.14_36.3_6.wav'
    wav, sr = sf.read(file_path, always_2d=True)
    wav = wav.transpose(1, 0)  # Ensure the shape is (channels, samples)
    print(np.max(wav), np.min(wav))
    print(wav.shape, sr)
    # wav2, sr = librosa.load(file_path, sr=sr)
    # print(np.max(wav2), np.min(wav2))
    # print(wav2.shape, sr)

    # wav_test = np.random.rand(wav.shape[1], wav.shape[0])

    spectrogram_augmentation = SpectrogramAugmentation()

    # spec_mag, phase = spectrogram_augmentation.get_spectrogram_phase(wav.transpose(1, 0))
    # print(spec_mag.shape, phase.shape)
    # spec_mag, phase = spectrogram_augmentation.get_spectrogram_phase(wav2)
    # print(spec_mag.shape, phase.shape)

    # Apply augmentations
    # spec_mag = spectrogram_augmentation.freq_spec_augment(spec_mag, num_freq_mask=2, freq_percentage=0.15)
    # print(spec_mag.shape)
    # spec_mag = spectrogram_augmentation.time_spec_augment(spec_mag, num_time_mask=2, time_percentage=0.15)
    # print(spec_mag.shape)

    wav, sample_rate, name = spectrogram_augmentation(wav, sr)
    print(wav.shape, sample_rate, name)

    spec, phase = spectrogram_augmentation.get_spectrogram_phase(wav)
    print(spec.shape, phase.shape)
    spec = spectrogram_augmentation.freq_spec_resize(spec, sample_rate=sample_rate, ratio=random.uniform(0.5, 1.5))
    print(spec.shape)
    spec = spectrogram_augmentation.time_spec_resize(spec, sample_rate=sample_rate, ratio=random.uniform(0.5, 1.5))
    print(spec.shape)

    # wav = librosa.istft(spec_mag * phase, hop_length=spectrogram_augmentation.hop_length, win_length=spectrogram_augmentation.win_length)
    # print(np.max(wav), np.min(wav))
    # print(wav.shape)
    # wav_tensor = torch.from_numpy(wav).float()
    # augmented_wav, sample_rate = spectrogram_augmentation.forward(wav_tensor, sr)
    # print(augmented_wav.shape, sample_rate)