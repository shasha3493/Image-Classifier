# Imports here
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

# Path for training, test and validation dataset
# data_dir = 'flowers'
# train_dir = data_dir + '/train'
# valid_dir = data_dir + '/valid'
# test_dir = data_dir + '/test'
train_dir = input('Enter the path to training dataset: ')
valid_dir = input('Enter the path to validation dataset: ')
test_dir = input('Enter the path to test dataset: ')

# TODO: Define your transforms for the training, validation, and testing sets

# Random Resizing,Cropping,Horizontal Flipping and Normalizing training images
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])

#Resizing,Center Cropping and Normalizing test images
test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])

#Resizing,Center Cropping and Normalizing validation images
valid_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])


# TODO: Load the datasets with ImageFolder
# image_datasets = datasets.ImageFolder(data_dir)
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms )
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms )

# TODO: Using the image datasets and the trainforms, define the dataloaders
# dataloaders = torch.utils.data.DataLoader(data_dir)
trainLoader = torch.utils.data.DataLoader(train_data,batch_size = 64, shuffle = True)
testLoader = torch.utils.data.DataLoader(test_data,batch_size = 64)
validLoader = torch.utils.data.DataLoader(valid_data,batch_size = 64)



# load in a mapping from category label to category name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

device = input("To run on GPU, enter 'cuda' otherwise enter 'cpu': ")
model_name = input("Choose one of the folloeing pretrained networks:\n1. VGG16\n2.ResNet-18\n")
lr = float(input("Enter learning rate (high values not recommended): "))
epochs = int(input("Enter number of epochs: "))
hidden_units = int(input('Enter number of hidden units: '))

# TODO: Build and train your network

device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')

# Loading pretrained network
if model_name == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_size = 25088
else:
    model = models.resnet18(pretrained=True)
    input_size = 512

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.05),
                                 nn.Linear(hidden_units,102), 
                                 nn.LogSoftmax(dim=1))
if model_name == 'vgg16':
    model.classifier = classifier
else:
    model.fc = classifier
# Loss function
criterion = nn.NLLLoss()

if model_name == 'vgg16':
    optimizer = optim.Adam(model.classifier.parameters(),lr=lr)
else:
    optimizer = optim.Adam(model.fc.parameters(),lr=lr)
    
model = model.to(device)

# Training Network


steps = 0

train_losses, valid_losses = [], []

# Running epochs
for e in range(epochs):
    model.train()
    running_loss = 0
    
    
    for images, labels in trainLoader:
        
#       Loading Data to GPU
        images,labels = images.to(device),labels.to(device)
    
        optimizer.zero_grad()
        
#       Feed Forward
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        
#       Back Propagation
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        valid_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in validLoader:
                
                # Loading Data to GPU
                images,labels = images.to(device),labels.to(device)
                
#               Output from Neural Network
                log_ps = model(images)
    
#               Keeping Track of Validation Loss
                valid_loss += criterion(log_ps, labels)
    
#               Calculating Output probabilities            
                ps = torch.exp(log_ps)
    
#               Finiding Class with highest probability and comparing it with the actual label to calculate accuracy
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

                
        train_losses.append(running_loss/len(trainLoader))
        valid_losses.append(valid_loss/len(validLoader))
        
#       Printing Results
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainLoader)),
              "Validation Loss: {:.3f}.. ".format(valid_loss/len(validLoader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(validLoader)))
  
    
    # TODO: Save the checkpoint 

checkpoint = {'input_size': input_size,
              'output_size': 102,
              'hidden_layers_units': hidden_units,
              'state_dict': model.state_dict(),
              'dropout':0.05,
              'class_to_idx': train_data.class_to_idx,
             'epochs': epochs,
             'optimizer_state':optimizer.state_dict,
             'arch':model_name,
             'hidden_layer':1,
             'learning_rate':lr}

torch.save(checkpoint, 'checkpoint.pth')

        