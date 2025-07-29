import torch
import torch.nn as nn
import soundfile as sf
import random
import os

class DataAugmentation(nn.Module):
    def __init__(self, augmentation_methods: list, random_weights=None):
        super(DataAugmentation, self).__init__()

        self.augmentation_methods = augmentation_methods
        self.num_transforms = len(augmentation_methods)

        self.transform_ids_list = [i for i in range(self.num_transforms)]
        self.random_weights = random_weights if random_weights is not None else [1 for _ in range(self.num_transforms)]
        assert len(self.transform_ids_list) == len(self.random_weights), "Length of transform_ids_list and random_weights must match"
    
    def forward(
            self, 
            wav, 
            sample_rate=48000, 
            mix_threshold=0.5
        ):
        wav = wav.copy()
        random_weights = self.random_weights.copy()

        transform_names = []
        transform_ids = []
        transform_id = random.choices(self.transform_ids_list, weights=random_weights, k=1)[0]
        transform_ids.append(transform_id)

        random_weights[transform_id] = 0  # Prevent re-selection of the same transform

        more_transform = random.random()
        while more_transform > mix_threshold and sum(random_weights) > 0:
            transform_id = random.choices(self.transform_ids_list, weights=random_weights, k=1)[0]
            transform_ids.append(transform_id)
            random_weights[transform_id] = 0  # Prevent re-selection of the same transform
            more_transform = random.random()

        for transform_id in transform_ids:
            wav, sample_rate, name = self.augmentation_methods[transform_id](wav, sample_rate=sample_rate, mix_threshold=mix_threshold)
            # print(wav.shape, sample_rate, name)
            transform_names.append(name)

        name = '_'.join(transform_names)

        return wav, sample_rate, name

    @staticmethod
    def read_wav_file(file_path):
        wav, sr = sf.read(file_path, always_2d=True)
        wav = wav.transpose(1, 0)  # Ensure the shape is (channels, samples)
        return wav, sr
    
    @staticmethod
    def write_wav_file(file_path, wav, sr):
        wav = wav.transpose(1, 0)  # Ensure the shape is (samples, channels)
        sf.write(file_path, wav, sr)

if __name__ == "__main__":
    from audio_augmentation import AudioAugmentation
    from spectrogram_augmentation import SpectrogramAugmentation

    file_path = './SPEAKER_00_1.14_36.3_6.wav'
    wav, sr = DataAugmentation.read_wav_file(file_path)
    print(wav.shape, sr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio_augmentation = AudioAugmentation(device=device)
    spectrogram_augmentation = SpectrogramAugmentation()

    augmentation_methods = [audio_augmentation, spectrogram_augmentation]
    random_weights = [0.7, 0.3]  # Example weights for audio and spectrogram augmentations
    data_augmentation = DataAugmentation(augmentation_methods=augmentation_methods, random_weights=random_weights)

    file_name = os.path.basename(file_path)
    augmented_wav, sample_rate, name = data_augmentation(wav, sr)
    new_file_name = file_name[:-4] + '_' + name + '.wav'
    DataAugmentation.write_wav_file(new_file_name, augmented_wav, sample_rate)
    print(augmented_wav.shape, sample_rate, name)