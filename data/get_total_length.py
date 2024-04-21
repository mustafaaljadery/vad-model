import os
import json

folder_path = "./temp_vad_output/"  

json_files = [file for file in os.listdir(folder_path) if file.endswith(".json")]
json_files = sorted(json_files)[:10]
data_list = []

for file_name in json_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r") as file:
        data = json.load(file)
        data_list.append(data)

count = 0

for data in data_list: 
    count += int(data["length_seconds"])
print(count/60/60)