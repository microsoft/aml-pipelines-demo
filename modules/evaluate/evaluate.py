from __future__ import print_function, division
import argparse
import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

def load_data(test_dir):
    ''' 
    Loads the the testing data 
    '''
    test_transform = transforms.Compose([
        transforms.Resize(200),
        transforms.CenterCrop(200),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.405],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    dataset_size = len(test_loader.dataset)
    class_names = test_dataset.classes
    
    return test_loader, dataset_size, class_names

def evaluate_model(model, criterion, dataloader, dataset_size, class_names, device):
    ''' 
    Evaluates the model 
    '''
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        corrects = torch.sum(preds == labels.data)
        running_corrects += corrects

    print('{}/{} predictions correct.'.format(running_corrects, dataset_size))
    loss = running_loss / dataset_size
    acc = running_corrects.double() / dataset_size
    print('Loss: {:.4f} Acc: {:.4f}'.format(loss, acc))
    
    return acc

# Define arguments
parser = argparse.ArgumentParser(description='Evaluate arg parser')
parser.add_argument('--test_dir', type=str, help='Directory where testing data is stored')
parser.add_argument('--model_dir', type=str, help='Directory where model is stored')
parser.add_argument('--accuracy_file', type=str, help='File to output the accuracy to')
args = parser.parse_args()

# Get arguments from parser
test_dir = args.test_dir
model_dir = args.model_dir
accuracy_file = args.accuracy_file

# Load testing data, model, and device
test_loader, dataset_size, class_names = load_data(test_dir)
model = torch.load(os.path.join(model_dir,'model.pt'))
device = torch.device('cuda:0')

# Define criterion
criterion = nn.CrossEntropyLoss()

# Evaluate model
acc = evaluate_model(model, criterion, test_loader, dataset_size, class_names, device)

# Output accuracy to file
with open(accuracy_file, 'w+') as f:
    f.write(str(acc.item()))
