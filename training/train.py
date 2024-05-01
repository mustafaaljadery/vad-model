from data_loader import AudioDataset
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch
import wandb

config = {
    "model_architecture": "MLP",
    "hours_of_audio": 3,
    "learning_rate": 0.001,
    "batch_size": 64,
    "dataset": "audio",
    "chunk_size": "10ms"
}

name = "small-mlp-1000-ms" 

wandb.init(project="vad-model", name=name)

train_dataset = AudioDataset(data_folder="./train_data/", frame_ms=1000)
val_dataset = AudioDataset(data_folder="./val_data/", frame_ms=1000)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class ClassificationMLP(nn.Module):
    def __init__(self, input_size=256, hidden_size= 256, num_classes=1):
        super(ClassificationMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ClassificationMLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

num_epochs = 5

def normalize_data(data):
    mean = torch.mean(data, dim=1, keepdim=True)
    std = torch.std(data, dim=1, keepdim=True)
    return (data - mean) / (std + 1e-8)

for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0

    for batch_idx, (data) in enumerate(train_dataloader):
        model.train()
        inputs, targets = data
        inputs = inputs.squeeze(1).to(device)
        targets = targets.to(device)

        soft_targets = torch.where(targets == 1, torch.tensor(0.95).to(device), torch.tensor(0.05).to(device))

        inputs = normalize_data(inputs)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs.squeeze(), soft_targets)
        loss.backward()
        optimizer.step()

        predicted = (outputs.squeeze() >= 0.5).float()
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        acc = correct / total
        
        running_loss += loss.item()
        running_acc += acc

        if (batch_idx + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                correct = 0
                total = 0
                false_positives = 0
                false_negatives = 0
                true_positives = 0
                true_negatives = 0

                all_targets = []
                all_predictions = []

                for inputs, targets in val_dataloader:
                    inputs = inputs.squeeze(1).to(device)
                    targets = targets.to(device).float() 
                    soft_targets = torch.where(targets == 1, torch.tensor(0.95).to(device), torch.tensor(0.05).to(device))
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), soft_targets)
                    val_loss += loss.item()
                    
                    predicted = (outputs.squeeze() >= 0.5).float() 
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                    false_positives += ((predicted == 1) & (targets == 0)).sum().item()
                    false_negatives += ((predicted == 0) & (targets == 1)).sum().item()
                    true_positives += ((predicted == 1) & (targets == 1)).sum().item()
                    true_negatives += ((predicted == 0) & (targets == 0)).sum().item()

                val_loss /= len(val_dataloader)
                val_acc = correct / total
                false_positive_rate = false_positives / (false_positives + true_negatives)
                false_negative_rate = false_negatives / (false_negatives + true_positives)

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"False Positive Rate: {false_positive_rate:.4f}, False Negative Rate: {false_negative_rate:.4f}")

                wandb.log({
                    "Val Loss": val_loss,
                    "Val Acc": val_acc,
                    "False Positive Rate": false_positive_rate,
                    "False Negative Rate": false_negative_rate
                })

            
            model.train()
    
    torch.save(model.state_dict(), f"./checkpoints/{name}-{epoch+1}.pth")