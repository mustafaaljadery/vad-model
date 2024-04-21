from datasets import Dataset, Audio
import os
import json

VAD_OUTPUT_FOLDER = "./temp_vad_output/"

def get_audio_files(folder_path):
    files = os.listdir(folder_path)
    mp3_files = [file for file in files if file.endswith(".mp3")]
    return sorted(mp3_files)[:25]

def get_json_files(folder_path):
    files = os.listdir(folder_path)
    json_files = [file for file in files if file.endswith(".json")]
    sorted_files = sorted(json_files)[:25]

    json_contents = []

    for json_file in sorted_files:
        file_path = os.path.join(folder_path, json_file)
    
        with open(file_path, "r") as file:
            json_data = json.load(file)
        
        json_contents.append(json_data)
    
    return json_contents

def parse_data(json_data):
    video_link = []
    title = []
    channel = []
    date = []
    original_language = []
    word_count = []
    length_seconds = []
    vad_output = []

    for data in json_data: 
        video_link.append(data["video_link"])
        title.append(data["title"])
        channel.append(data["channel"])
        date.append(data["date"])
        original_language.append(data["original_language"])
        word_count.append(data["word_count"])
        length_seconds.append(data["length_seconds"])
        vad_output.append(data["vad_output"])
    
    return video_link, title, channel, date, original_language, word_count, length_seconds, vad_output

if __name__ == '__main__':
    audio_files = get_audio_files(VAD_OUTPUT_FOLDER)
    audio_paths = [VAD_OUTPUT_FOLDER + x for x in audio_files]
    json_data = get_json_files(VAD_OUTPUT_FOLDER)
    video_link, title, channel, date, original_language, word_count, length_seconds, vad_output = parse_data(json_data)
    audio_dataset = Dataset.from_dict({
        "audio": audio_paths, 
        "video_link": video_link,
        "title": title, 
        "channel": channel,
        "date": date,
        "original_language": original_language,
        "word_count": word_count,
        "length_seconds": length_seconds,
        "vad_output": vad_output
    }).cast_column("audio", Audio())
    audio_dataset.push_to_hub("mustafaaljadery/youtube_commons_vad_25_sample")
