import matplotlib.pyplot as plt
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection 
import torch

model = Model.from_pretrained("pyannote/segmentation-3.0")
pipeline = VoiceActivityDetection(segmentation=model)

params = {
    "min_duration_on": 0.0, 
    "min_duration_off": 0.0
}

device = torch.device("mps")
pipeline.instantiate(params)
pipeline.to(device)

def visualize_voice_activity(segments):
    start_times = []
    end_times = []

    for segment in segments:
        start_times.append(segment['start'])
        end_times.append(segment['end'])

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_title('Voice Activity Segments')
    ax.set_xlabel('Time')
    ax.set_yticks([])

    for i in range(len(start_times)):
        ax.hlines(0, start_times[i], end_times[i], linewidth=8, color='blue')

    ax.grid(True)
    plt.tight_layout()
    plt.show()


vad_data = []

# replace the audio here with the test audios
vad = pipeline("./data_review_audio/audio-5.mp3")

for item in vad.get_timeline().support(): 
    vad_data.append({
        'start': item.start,
        'end': item.end
    })

visualize_voice_activity(vad_data)