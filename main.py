
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
import torchtext
from torch.optim import Adam

from torch.utils.data import DataLoader
from torch.autograd import Variable

import csv

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn as nn

from PIL import Image
import numpy as np




#mapping from label to integer and vice versa
categoryToNum={'scarves': 0, 'flip flops': 1, 'topwear': 2, 'sandal': 3, 'bags': 4, 'socks': 5, 'shoes': 6, 'watches': 7, 'dress': 8, 'headwear': 9, 'jewellery': 10, 'bottomwear': 11, 'innerwear': 12, 'wallets': 13, 'belts': 14, 'saree': 15, 'nails': 16, 'loungewear and nightwear': 17, 'lips': 18, 'eyewear': 19, 'makeup': 20, 'ties': 21, 'fragrance': 22, 'cufflinks': 23, 'free gifts': 24, 'apparel set': 25, 'accessories': 26}
inv_map={0: 'Scarves', 1: 'Flip flops', 2: 'Jewellery', 3: 'Bottomwear', 4: 'Innerwear', 5: 'Wallets', 6: 'Belts', 7: 'Saree', 8: 'Nails', 9: 'Loungewear and Nightwear', 10: 'Lips', 11: 'Eyewear', 12: 'Topwear', 13: 'Makeup', 14: 'Ties', 15: 'Fragrance', 16: 'Cufflinks', 17: 'Free Gifts', 18: 'Apparel Set', 19: 'Accessories', 20: 'Sandal', 21: 'Bags', 22: 'Socks', 23: 'Shoes', 24: 'Watches', 25: 'Dress', 26: 'Headwear'}



#store the csv file's data
train_inputs = np.genfromtxt('myData/train.csv', delimiter=',')
test_inputs = np.genfromtxt('myData/test.csv', delimiter=',')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load the images
train_transformations = transforms.Compose([


    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_set = datasets.ImageFolder("myData/trainImages", transform = train_transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

#CNN implementation
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.convolution = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.convolution(input)
        output = self.bn(output)
        output = self.relu(output)

        return output
class CNN(nn.Module):
    def __init__(self, NUM_CLASS = 27):
        super(CNN, self).__init__()
        self.unit1 = Unit(in_channels=3, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)


        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = m = nn.Dropout(p=0.2)

        self.unit3 = Unit(in_channels=32, out_channels=64)
        self.unit4 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = m = nn.Dropout(p=0.2)



        self.fc = nn.Linear(in_features=20*15*64, out_features=NUM_CLASS)

    def forward(self, input):
        output = self.unit1(input)
        output = self.unit2(output)
        output = self.pool1(output)
        output = self.drop1(output)

        output = self.unit3(output)
        output = self.unit4(output)
        output = self.pool2(output)
        output = self.drop2(output)

        output = output.view(-1,20*15*64)

        output = self.fc(output)

        return output

NUM_CLASS = 27
NUM_EPOCH=10
model = CNN(NUM_CLASS).to(device)
model.cuda()
optimizer = Adam(model.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss().to(device)





def train(num_epochs):


    for epoch in range(NUM_EPOCH):
        model.train()
        train_acc = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to gpu if available

            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        train_acc = train_acc /len(train_set)
        train_loss = train_loss / len(train_set)
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} ".format(epoch, train_acc, train_loss,))



train(NUM_EPOCH)
model.eval()


#we finished training model, now predict the test set
def predict(path):

    image = Image.open(path)
    transformation = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    imageTensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    imageTensor = imageTensor.unsqueeze_(0)


    #imageTensor=Variable(imageTensor.cuda())

    input = Variable(imageTensor.cuda())
    output = model(input)
    index = output.data.argmax()
    return index


with open ('submission.csv', 'w', newline='') as myFile:
    print('writing')
    writer=csv.writer(myFile)
    writer.writerow(['id', 'category'])

    for k in range(len(test_inputs) - 1):

        file_name = 'myData/shuffled-images/' + str((int)(test_inputs[k + 1][0])) + '.jpg'
        writer.writerow([(int)(test_inputs[k + 1][0]), inv_map[(int)(predict(file_name))]])

