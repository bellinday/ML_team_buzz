import matplotlib.pyplot as plt
import torch
import cv2
import torch.nn as nn
import torchvision.transforms.functional as fn
import numpy
import torchvision
from IPython.core.pylabtools import figsize
from torch.utils import data
from torchvision import transforms

# steps involved:
# 1. Download and define the training, testing data
# 2. Define the model by creating neural networks
# 3. Calculate the gradient, loss and update the model

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

transformation = torchvision.transforms.ToTensor()
test_transform  = torchvision.transforms.Compose([
    transforms.Resize((784)),
    transforms.ToTensor(),
])
num_class = 10
input_size = 784
hidden_size = 100
batches = 100
learning_rate = 0.01
num_epochs = 1

mnist_train = torchvision.datasets.MNIST(root="./data", train=True, transform=transformation, download=True)
mnist_test = torchvision.datasets.MNIST(root="./data", train=False, transform=transformation, download=False)

train_data = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batches, shuffle=True)
test_data = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batches, shuffle=False)

examples = iter(train_data)
samples, label = examples.__next__()
print(samples.shape, label.shape)

test_examples = iter(test_data)
test_samples, test_lables = test_examples.__next__()


# for i in range(6):
#     plt.subplot(3,3, i+1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()


# plt.imshow(samples[99][0], cmap='gray')
# plt.show()


# 2. Create the neural network layers
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNetwork(input_size, hidden_size, num_class)


# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 4. Training loop
n_steps = len(train_data)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_data):
# reshape the images
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
# forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

# backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1)%100 ==0 :
            print(f"epoch {epoch+1} loss = {loss.item()}")

# test and evaluate

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_data:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

# value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += label.shape[0]
        n_correct = (predictions == labels).sum().item()

    acc = 100.0* n_correct / n_samples
    print(f"accuracy = {acc}")

# cap = cv2.VideoCapture(0)
# while True:
#     status, photo = cap.read()
#     cv2.imshow('Attention please', photo)
#     if cv2.waitKey(1) == 13:
#         img = cv2.imwrite("new.png", photo)
#         break
# cv2.destroyAllWindows()

test_image = test_samples[74][0]
test_image = test_image.reshape(-1, 784).to(device)
# images = torch.from_numpy(test_image)
# width, height = test_image.size
# crop = test_image.resize((784, int(784*(height/width))) if width < height else (int(784*(width/height)), 784))
# crop = fn.center_crop(images, output_size=[28])
# images = torchvision.transforms.CenterCrop(size=784)
# images = images.reshape(-1, 28 * 28).to(device)
# print(type(crop))
# print(crop.shape)
print(model(test_image))
plt.imshow(test_samples[74][0], cmap='gray')
plt.show()