import torch
import os
import shutil

from data_augmentation import DataAugmentation
from audio_augmentation import AudioAugmentation
from spectrogram_augmentation import SpectrogramAugmentation

if __name__ == '__main__':
    wav_file_path = './SPEAKER_00_1.14_36.3_6.wav'
    label_file_path = '.' # Please fill this
    output_folder_path = './output'
    os.makedirs(output_folder_path, exist_ok=True)

    wav, sr = DataAugmentation.read_wav_file(wav_file_path)
    # print(wav.shape, sr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio_augmentation = AudioAugmentation(device=device)
    spectrogram_augmentation = SpectrogramAugmentation()

    augmentation_methods = [audio_augmentation, spectrogram_augmentation]
    random_weights = [0.7, 0.3]  # Example weights for audio and spectrogram augmentations
    data_augmentation = DataAugmentation(augmentation_methods=augmentation_methods, random_weights=random_weights)

    wav_file_name = os.path.basename(wav_file_path)
    label_file_name = os.path.basename(label_file_path)
    augmented_wav, sample_rate, name = data_augmentation(wav, sr)
    new_file_name = wav_file_name[:-4] + '_' + name + '.wav'
    label_file_name = label_file_name[:-4] + '_' + name + '.txt'
    output_wav_path = os.path.join(output_folder_path, new_file_name)

    DataAugmentation.write_wav_file(output_wav_path, augmented_wav, sample_rate)
    print(augmented_wav.shape, sample_rate, name)
    
    # Copy label file to output folder
    output_label_path = os.path.join(output_folder_path, label_file_name)
    shutil.copy(label_file_path, output_label_path)

