import torch
from torch import nn
import torch.nn.functional as F
from data_loader import AudioDataset
from torch.utils.data import DataLoader

dataset = AudioDataset(data_folder="./test_data/", frame_ms=1000)
dataloader = DataLoader(dataset, batch_size=32)

class ClassificationMLP(nn.Module):
    def __init__(self, input_size=256, hidden_size= 256, num_classes=2):
        super(ClassificationMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class ClassificationMLP2(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, num_classes=2):
        super(ClassificationMLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1_drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2_drop = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc3_drop = nn.Dropout(p=0.2)
        self.last = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1((self.fc1(x))))
        out = F.relu(self.bn2((self.fc2(out))))
        out = F.relu(self.bn3((self.fc3(out))))
        out = self.last(out)
        return out

device = "mps"
model = ClassificationMLP().to(device)
model.load_state_dict(torch.load('./checkpoints/small-mlp-1000-ms-5.pth'))
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs = inputs.squeeze(1).to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted_labels == targets).sum().item()
        print(targets, predicted_labels)


    accuracy = 100 * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")