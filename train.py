import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader import SuperSlowMoDataset
import model 

from torch.utils.data import DataLoader, Dataset

data_dir = "/home/quimisagi/Daigaku/Datasets/adobe240/train"
train_dataset = SuperSlowMoDataset(data_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    num_workers = torch.cuda.device_count() * 4
else:
    num_workers = 4

LEARNING_RATE = 3e-4
BATCH_SIZE = 64

flowCompNet = model.UNet(6, 4)
flowCompNet = flowCompNet.to(device)
flowCompNet.train()

optimizer = optim.Adam(flowCompNet.parameters(), lr=LEARNING_RATE)

train_dataloader = DataLoader(dataset=train_dataset,
                              num_workers=num_workers, pin_memory=False,
                              batch_size=BATCH_SIZE,
                              shuffle=True)


# Main training loop

losses = []

EPOCHS = 10
for epoch in  tqdm(range(EPOCHS)):
    print(f"Epoch {epoch+1}")
    train_running_loss = 0.0
    for index, triplet in enumerate(train_dataloader):
        flowCompNet.zero_grad()
        frame0 = triplet[0].to(device)
        frame1 = triplet[1].to(device)
        frame2 = triplet[2].to(device)
        flow = flowCompNet(torch.cat((frame0, frame2), 1))
        loss = nn.L1Loss()(flow, frame1)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()
    losses.append(train_running_loss)
    print(f"Loss: {train_running_loss}")
