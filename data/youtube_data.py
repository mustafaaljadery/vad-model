### Process: 
# 1. Randomly sample 25K entries based on the dataset from huggingface. 
# 2. Download the videos into the `temp_videos` folder with a json document of same name about the details of the video.
# 3. (different file) VAD model runs on this data.

from pytube import YouTube
from moviepy.editor import *
from datasets import load_dataset
import os
import json
import time

def get_dataset_samples():
    dataset_name = "PleIAs/YouTube-Commons"
    split = "train"
    data_files = {"train": [f"cctube_{i}.parquet" for i in range(12)]}
    dataset = load_dataset(dataset_name, data_files=data_files, split=split)
    return dataset.shuffle(seed=527423).take(50000)

def download_youtube_audio(url, temp_path, output_path):
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_file = audio_stream.download(output_path=temp_path)
        
        video = AudioFileClip(audio_file)
        wav_file = output_path 
        video.write_audiofile(wav_file)
        os.remove(audio_file)
        
        video_length_seconds = video.duration  # Get the duration of the video in seconds
        return video_length_seconds
    except Exception as e:
        return False

def file_in_folder(folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    return os.path.isfile(file_path)


def save_dict_to_json(dict, folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, 'w') as json_file:
        json.dump(dict, json_file)

VIDEO_OUTPUT_DIR = "./temp_videos/"
VAD_OUTPUT_DIR = "./temp_vad_output/"

if __name__ == '__main__':
    data = get_dataset_samples()

    for i in range(len(data)): 
        if file_in_folder(VAD_OUTPUT_DIR, f"audio-{i}.json") or file_in_folder(VIDEO_OUTPUT_DIR, f"audio-{i}.wav"):
            continue

        video = data[i]

        json_data = {
            "video_link": video["video_link"],
            "title": video["title"],
            "channel": video["channel"],
            "date": video["date"],
            "original_language": video["original_language"],
            "word_count": video["word_count"]
        }

        download_data = download_youtube_audio(video["video_link"], VIDEO_OUTPUT_DIR, f"{VIDEO_OUTPUT_DIR}audio-{i}.wav")

        if download_data: 
            json_data["length_seconds"] = download_data
            save_dict_to_json(json_data, VIDEO_OUTPUT_DIR, f"audio-{i}.json")
            time.sleep(2)