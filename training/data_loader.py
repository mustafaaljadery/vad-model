from signals import Signal
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json

class AudioDataset(Dataset):
    def __init__(self, data_folder, frame_ms):
        self.data_folder = data_folder
        self.frame_ms = frame_ms
        self.data = []

        files = self.get_all_files() 
        for audio_data in files: 
            audio_file = audio_data['audio']
            metadata_file = audio_data['metadata']

            waveforms = self.get_input_waveform_and_sample_rate(f"{self.data_folder}{audio_file}")
            labels = self.get_output_list(f"{self.data_folder}{metadata_file}", len(waveforms))

            for i in range(len(waveforms)):
                self.data.append((waveforms[i], torch.tensor(labels[i])))

            # self.data.append(waveforms, torch.tensor(labels))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_value, output_value = self.data[index]
        return input_value, output_value
    
    def get_all_files(self):
        json_files = [file for file in os.listdir(self.data_folder) if file.endswith(".json")]

        return [{'audio': x.split(".json")[0] + ".mp3", 'metadata': x} for x in json_files]

    def get_input_waveform_and_sample_rate(self, audio_path):
        y, sr = librosa.load(audio_path, sr=16000)
        y_tensor = torch.from_numpy(y).float()
        sig = Signal(y_tensor, sr)
        frame_size = int(sig.sample_rate * (self.frame_ms / 1000.0))
        X = sig.get_MFCC(hop_length=frame_size, n_mels=256, n_mfcc=256).transpose(2,0).transpose(1,2)
        return X 

    def get_output_list(self, json_file, waveform_length):
        with open(json_file) as f:
            data = json.load(f)
        vad_output = data["vad_output"]

        interval = (1000 / self.frame_ms)

        audio_presence_list = [0] * waveform_length 

        for start_end in vad_output:
            start_time, end_time = start_end

            start_index = int(start_time * interval)
            end_index = int(end_time * interval)

            for i in range(start_index, end_index + 1):
                if i < waveform_length:
                    audio_presence_list[i] = 1

        return audio_presence_list