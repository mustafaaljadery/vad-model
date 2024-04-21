import time
import torch
import torchaudio
import json
import os
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

model = Model.from_pretrained(
  "pyannote/segmentation-3.0")

pipeline = VoiceActivityDetection(segmentation=model)

HYPER_PARAMETERS = {
  "min_duration_on": 0.0,
  "min_duration_off": 0.0
}

device = torch.device("mps")
pipeline.instantiate(HYPER_PARAMETERS)
pipeline.to(device)

TEMP_VIDEOS_FOLDER = "./temp_videos/"
VAD_OUTPUT_FOLDER = "./temp_vad_output/"

# Getting the oldest wav file to avoid race condition of pytube conversion.
def get_oldest_wav_file(folder_path):
    wav_files = [file for file in os.listdir(folder_path) if file.endswith(".wav")]
    
    if not wav_files:
        return None
    
    oldest_file = min(wav_files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
    return os.path.join(folder_path, oldest_file), oldest_file

def count_wav_files(folder_path):
    wav_files = [file for file in os.listdir(folder_path) if file.endswith(".wav")]
    return len(wav_files)

def get_json_data(json_path):
    with open(json_path, 'r') as json_file:
        return json.load(json_file)

def save_dict_to_json(dict, folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, 'w') as json_file:
        json.dump(dict, json_file)

def save_mp3_file(file_path, output_path):
    waveform, sample_rate = torchaudio.load(file_path)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    torchaudio.save(output_path, waveform, 16000, format="mp3")

def vad_data_to_dict(file_path, video_link, title, channel, date, original_language, word_count, length_seconds):
    temp_dict = {
        "video_link": video_link, 
        "title": title,
        "channel": channel, 
        "date": date, 
        "original_language": original_language,
        "word_count": word_count,
        "length_seconds": length_seconds 
    }
    vad = pipeline(file_path)
    speech_times = []
    for speech in vad.get_timeline().support():
        speech_times.append([
            speech.start, speech.end
        ])

    temp_dict["vad_output"] = speech_times
    return temp_dict 

if __name__ == '__main__':
    num_wav_pending = count_wav_files(TEMP_VIDEOS_FOLDER)
    while (num_wav_pending):
        file_path, file_name = get_oldest_wav_file(TEMP_VIDEOS_FOLDER)
        json_path = file_path.split(".wav")[0] + ".json"
        output_name = file_name.split(".wav")[0] + ".json"
        mp3_output = VAD_OUTPUT_FOLDER + file_name.split(".wav")[0] + ".mp3"

        json_data = get_json_data(json_path)

        vad_data = vad_data_to_dict(file_path, json_data['video_link'], json_data['title'], json_data['channel'], json_data['date'], json_data['original_language'], json_data['word_count'], json_data['length_seconds'])

        save_dict_to_json(vad_data, VAD_OUTPUT_FOLDER, output_name)
        save_mp3_file(file_path, mp3_output)

        os.remove(json_path)
        os.remove(file_path)

        num_wav_pending = count_wav_files(TEMP_VIDEOS_FOLDER)