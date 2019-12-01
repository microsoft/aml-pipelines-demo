from __future__ import print_function, division
import argparse
import time
import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import warnings
warnings.filterwarnings("ignore")

def load_data(train_dir, valid_dir, batch_size):
    '''
    Loads the data using Pytorch dataset loader
    '''
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(200),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.405],
                             std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(200),
        transforms.CenterCrop(200),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.405],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transform)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    dataset_sizes = { 'train': len(train_loader.dataset), 'valid': len(valid_loader.dataset) }

    class_names = train_dataset.classes
    
    return dataloaders, dataset_sizes, class_names

def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, class_names, device):
    '''
    Train the model
    '''
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                corrects = torch.sum(preds == labels.data).float()
                running_corrects += corrects

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

 # Define arguments
parser = argparse.ArgumentParser(description='Training arg parser')
parser.add_argument('--train_dir', type=str, help='Directory where training data is stored')
parser.add_argument('--valid_dir', type=str, help='Directory where validation data is stored')
parser.add_argument('--output_dir', type=str, help='Directory to output the model to')
parser.add_argument('--num_epochs', type=int, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, help='Train batch size')
parser.add_argument('--learning_rate', type=float, help='Learning rate of optimizer')
parser.add_argument('--momentum', type=float, help='Momentum of optimizer')
args = parser.parse_args()

# Get arguments from parser
train_dir = args.train_dir
valid_dir = args.valid_dir
output_dir = args.output_dir
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
momentum = args.momentum

# Load datasets
dataloaders, dataset_sizes, class_names = load_data(train_dir, valid_dir, batch_size)

# Load pretrained resnet model
resnet = models.resnet18(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, len(class_names)) 

# Use CUDA if it is available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
resnet = resnet.to(device)

# Specify criterion
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(resnet.parameters(), lr=learning_rate, momentum=momentum)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train model
model = train_model(resnet, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, class_names, device)

# Save model
print('Saving model')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
torch.save(model, os.path.join(output_dir, 'model.pt'))
classes_file = open(os.path.join(output_dir, 'class_names.pkl'), 'wb')
pickle.dump(class_names, classes_file)
classes_file.close()