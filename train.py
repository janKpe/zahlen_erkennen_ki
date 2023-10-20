import torch.nn.functional as F
import torch
from torchvision import transforms
from os import listdir
from PIL import Image
import random
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from os import path

device = "cuda"
batchs = 390
batchsize = 64
epochs = int(input("Epochs: ")) 
learingRate = input("Learning rate: ")

if learingRate == "":
    learingRate = 0.001
else:
    learingRate = float(learingRate)



train_data_list = []


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


trans = transforms.Compose([
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor(),
                            normalize])

def getData():
    for batch in range(batchs):
        target_list = []
        train_data = []
        for count in range(batchsize):
             
            num = random.choice(listdir("MNIST Dataset JPG format\MNIST - JPG - training"))
            list = []
            for e in range(10):
                if int(num) == e:
                    list.append(1)
                else:
                    list.append(0)

            

            img = random.choice(listdir(f"MNIST Dataset JPG format\MNIST - JPG - training\{num}"))
            img = Image.open(f"MNIST Dataset JPG format\MNIST - JPG - training\{num}\{img}")
            img = trans(img)

            train_data.append(torch.Tensor(img).to(device))
            target_list.append(list)


            print("Loading Image {}/{} \tBatch: {}/{} \tPercentage Done: {}%".format(count, batchsize, batch, batchs, round(100*len(train_data_list)/batchs, 2)))

        train_data_list.append((torch.stack(train_data), (target_list)))

getData()    


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=4)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=4)

        self.fc1 = nn.Linear(48, 10)

        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)  
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)  
        x = F.relu(x)

        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x

if path.isfile("Netz.pt"):
    model = torch.load("Netz.pt")
    model.to(device)
    
else:
    model = Netz().to(device)


criterion = F.binary_cross_entropy
optimizer = optim.Adam(model.parameters(), lr=learingRate)
loss_plot = []


def train():
    model.train()
    for epoch in range(epochs):
        index = 0
        for data, target in train_data_list:

            out = model(data)
            target = torch.Tensor(target).to(device)

            loss = criterion(out, target) 
            loss_plot.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            torch.save(model, "Netz.pt")

            print("Train Epoch: " + str(epoch) + "/" + str(epochs), "[" + str(index) + "/" + str(batchs) + "]", "\tLoss: {:.6f}".format(loss.item()))

            index += 1



train()


plt.plot(loss_plot)
plt.show()
