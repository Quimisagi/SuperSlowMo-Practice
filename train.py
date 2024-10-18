import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader import SuperSlowMoDataset
import model 
import torchvision

from torch.utils.data import DataLoader, Dataset

data_dir = "/home/quimisagi/Daigaku/Datasets/adobe240/train"
train_dataset = SuperSlowMoDataset(data_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    num_workers = torch.cuda.device_count() * 4
else:
    num_workers = 4

LEARNING_RATE = 0.0001
BATCH_SIZE = 6
progress_iter = 250
checkpoint_epoch = 1

flowCompNet = model.UNet(6, 4)
flowCompNet = flowCompNet.to(device)
flowCompNet.train()

arbitaryTimeFlowInterpolator = model.UNet(20, 4)

backwarp = model.Backwarp(512, 512, device)

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3.to(device)
for param in vgg16_conv_4_3.parameters():
		param.requires_grad = False

L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()
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
        F_0_1 = flow[:,:2,:,:]
        F_1_0 = flow[:,2:,:,:]

        flowCoefficients = model.GetFlowCoefficients(device)
        F_t_0 = flowCoefficients[0] * F_0_1 + flowCoefficients[1] * F_1_0
        F_t_1 = flowCoefficients[2] * F_0_1 + flowCoefficients[3] * F_1_0

        g_I0_F_t_0 = backwarp(frame0, F_t_0)
        g_I1_F_t_1 = backwarp(frame2, F_t_1)

        intrp = torch.cat((frame0, frame1, frame2, g_I0_F_t_0, g_I1_F_t_1, F_t_0, F_t_1), 1)

        F_t_0_f = intrp[:, :2, :, :]
        F_t_1_f = intrp[:, 2:, :, :]

        g_I0_F_t_0_f = backwarp(frame0, F_t_0_f)
        g_I1_F_t_1_f = backwarp(frame2, F_t_1_f)

        warpCoefficients = model.GetWarpCoefficients(device)

        Ft_p = (warpCoefficients[0] * g_I0_F_t_0_f + warpCoefficients[1] * g_I1_F_t_1_f)

        # Loss
        recnLoss = L1_lossFn(Ft_p, frame1)
        
            
        prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(frame1))
        
        warpLoss = L1_lossFn(g_I0_F_t_0, frame1) + L1_lossFn(g_I1_F_t_1, frame1) + L1_lossFn(backwarp(frame0, F_1_0), frame1) + L1_lossFn(backwarp(frame1, F_0_1), frame0)
        loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss

        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()
    losses.append(train_running_loss)
    print(f"Loss: {train_running_loss}")
    if epoch % checkpoint_epoch == 0:
        torch.save(flowCompNet.state_dict(), f"flowCompNet_{epoch}.pth")
        torch.save(arbitaryTimeFlowInterpolator.state_dict(), f"arbitaryTimeFlowInterpolator_{epoch}.pth")
        torch.save(optimizer.state_dict(), f"optimizer_{epoch}.pth")
        print("Model saved")
