
# Imports here
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json


# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)
    classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers_units']),
                                 nn.ReLU(),
                                 nn.Dropout(checkpoint['dropout']),
                                 nn.Linear(checkpoint['hidden_layers_units'],checkpoint['output_size']), 
                                 nn.LogSoftmax(dim=1))
    
#     vgg16 net has attribute 'classifier' while resnet18 has 'fc'
    if checkpoint['arch'] == 'vgg16':
        model.classifier = classifier
    else:
        model.fc = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    
#   Resizing Image keeping Aspect Ratio same  
    width, height = im.width, im.height
    aspect_ratio = width / height
    if width > height:
        new_height = 256
        new_width = int(aspect_ratio * new_height)
    elif height > width:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 256
        new_width = 256
    im = im.resize((new_width, new_height))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1)

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    device = input("To use GPU for predictions, enter 'cuda' otherwise enter 'cpu': ")
    device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    im=process_image(image_path)
    im = torch.from_numpy(im).type(torch.FloatTensor)

    im = im.unsqueeze(0)
    im=im.to(device)
    ps = torch.exp(model.forward(im))
    top_p, top_class = ps.topk(topk, dim=1)
    return(top_p,top_class)

# TODO: Display an image along with the top 5 classes
import matplotlib.pyplot as plt

# Loading the model
path = input('Enter the path to checkpoint file to load from: ')
model = load_checkpoint(path)

# Opening and Displaying the image
image_path = input('Enter path of image: ')
topk=int(input('Enter number of top classes for which you want to see the probability: '))
class_name = input('Do you want to see class name(y/n)? ')
if class_name == 'y':
    mapping = input('Enter the json file name : ')

# Feed Forward
probs, classes = predict(image_path, model,topk)


# Calculating the probabilties of top 5 names of flowers
probs = list(probs.detach().to('cpu').numpy()[0,:])
classes = list(classes.detach().to('cpu').numpy()[0,:])
print('Classes: ',classes)
print('Associated probabilities: ',probs)

if class_name == 'y':

    # load in a mapping from category label to category name
    with open(mapping, 'r') as f:
        cat_to_name = json.load(f)

    names = []
    for i in range(topk):
        label = classes[i]
        for cl,idx in model.class_to_idx.items():
            if idx==label:
                for cat,name in cat_to_name.items():
                    if cl==cat:
                        names.append(name)
    print('Names of Flower: ',names)
    print('Associated probabilities: ',probs)

    

