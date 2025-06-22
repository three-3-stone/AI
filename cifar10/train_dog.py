from cnn import Net
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns

CFAIR10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# load a image
image = Image.open('./sample.jpg')

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

image_transformed = transform(image)
# print(image_transformed.size())

net = Net() 
net.load_state_dict(torch.load('./model/sche_best_model_cifar10.pth', weights_only=True))

image_transformed = image_transformed.unsqueeze(0)
output = net(image_transformed)
predict_value, predict_idx = torch.max(output, 1)  # 求指定维度的最大值，返回最大值以及索引

plt.figure()
plt.imshow(np.array(image))
plt.title(CFAIR10_names[predict_idx])
plt.axis('off')

plt.show()
