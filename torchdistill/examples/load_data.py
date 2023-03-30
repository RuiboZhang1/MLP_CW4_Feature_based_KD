import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torchvision import datasets, transforms

# Define the transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070754, 0.48655024, 0.44091907), (0.26733398, 0.25643876, 0.2761503)),
])

# Load the CIFAR-100 training data
train_data = datasets.CIFAR100(root='./resource/dataset/cifar100', train=True, download=False, transform=transform)

# Split the training data into train and validation sets while maintaining proportional class distribution
train_indices = []
val_indices = []
labels = np.array(train_data.targets)

for i in range(100):
    # Get indices of examples with class i
    indices = np.where(labels == i)[0]
    
    # Split the indices into train and validation sets
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    train_index, val_index = next(splitter.split(np.zeros(len(indices)), indices))
    
    # Add the indices to the appropriate lists
    train_indices.extend(indices[train_index][:250])
    val_indices.extend(indices[val_index][:250])
    
# Create samplers for the train and validation sets
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

# Create data loaders for the train and validation sets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_data, batch_size=32, sampler=val_sampler)

print(train_loader)
print(val_loader)