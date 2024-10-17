import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SuperSlowMoDataset(Dataset):
    def __init__(self, dataset_root):
        self.frames_path = self._load_frames(dataset_root)
        self.triplets = self.generate_triplets(self.frames_path)
        self.frames = self.randomize_triplets(self.triplets)
        
        # Define the transformation
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    
    def _load_frames(self, path):
        frames_path = []
        if not os.path.exists(path):
            raise Exception(f'{path} does not exist')
        sorted_folders = sorted(os.listdir(path))
        for index, folder in enumerate(sorted_folders):
            frames_path.append([])
            sorted_images = sorted(os.listdir(os.path.join(path, folder)))
            for image in sorted_images:
                frames_path[index].append(os.path.join(path, folder, image))
        return frames_path
    
    def generate_triplets(self, frames_path):
        triplets = set()
        unique_triplets = []
        
        for clip in frames_path:
            for i in range(len(clip) - 2):
                triplet = (clip[i], clip[i+1], clip[i+2])
                if triplet not in triplets:
                    triplets.add(triplet)
                    unique_triplets.append(triplet)

        return unique_triplets

    def randomize_triplets(self, triplets):
        random.shuffle(triplets)
        return triplets

    def __len__(self):
        # Return the total number of triplets
        return len(self.frames)

    def __getitem__(self, idx):
        # Get the triplet paths
        triplet_paths = self.frames[idx]

        # Load and transform each frame in the triplet
        frame0 = Image.open(triplet_paths[0])
        frame1 = Image.open(triplet_paths[1])
        frame2 = Image.open(triplet_paths[2])

        # Apply transformations
        frame0 = self.transform(frame0)
        frame1 = self.transform(frame1)
        frame2 = self.transform(frame2)

        # Return the triplet of transformed frames
        return frame0, frame1, frame2

