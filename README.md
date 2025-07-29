# AudioAugmentation
audio_augmentation.py: Biến đổi trực tiếp trên raw audio waveform  
spectrogram_augmentation.py: Biến đổi trên miền spectrogram của wavform.  
data_augmentation.py: Dùng cả hai phương pháp augmentation trên.  
Lưu ý: 
- Sử dụng phương thức read_wav_file trong file data_augmentation.py để load file đúng format và đưa vào các phương pháp augmentation.  
- Ngoài ra, có thể dùng lẻ các phương pháp augmentation.  
- Ngoài torch thì cần cài thêm một số thư viện sau: librosa, soundfile, torch_audiomentations.