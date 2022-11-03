import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f

#dummy varible assigned now but will take the number of classes per dataset
classes = 120

model = torchvision.models.resnet18(pretrained = False) #flag to either load the weights or with random initialization

#freeze model params
#uncomment for transfer learning (right now we're not in-order to train from scratch)
for param in model.parameters():
    param = param.requires_grad_(False)
    
#new layer
model.fc = nn.Sequential(
                      nn.Linear(model.fc.in_features, 512),
                      nn.ReLU(),
                      nn.Dropout(0.4),
                      nn.Linear(512,256),
                      nn.ReLU(),
                      nn.Dropout(0.2),
                      nn.Linear(256, len(classes)),                   
                      nn.LogSoftmax(dim=1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device) #Moving the model to GPU                      


#Additional info - details on the training loop 
"""
model, history = train_loop(
    model,
    criterion,
    optimizer,
    trainloader,
    validloader,
    save_model_name="./resnet18_fish_scratch.pt",
    max_epochs_stop=100,
    num_epochs=100,
    num_epochs_report=3)
"""