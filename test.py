import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import random
from os import listdir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


trans = transforms.Compose([
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor(),
                            normalize])


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=4)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=4)
        self.fc1 = nn.Linear(80, 10)
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


model = torch.load("Netz.pt").to("cuda")


while True:
    model.eval()
    num = random.choice(listdir("MNIST Dataset JPG format\MNIST - JPG - testing"))

    img = random.choice(listdir(f"MNIST Dataset JPG format\MNIST - JPG - testing\{num}"))
    img = Image.open(f"MNIST Dataset JPG format\MNIST - JPG - testing\{num}\{img}")
    img_in = trans(img).to("cuda") 
    img_in.unsqueeze_(0)

    plt.imshow(img, cmap='viridis')
    out = model(img_in).tolist()[0]

    biggest = -500
    for e in out:
        
        if e > biggest:
            biggest = e
    plt.title(f"prediction: {str(out.index(biggest))} actual: {num}")
    plt.show()

