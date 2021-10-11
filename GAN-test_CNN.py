import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

# define the class for the CNN-classifier
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(      # input shape: (batch_size, 1, 28, 28)
            nn.Conv2d(1, 16, 5, 1, 2),   # shape: (batch_size, 16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),             # shape: (batch_size, 16, 14, 14)
            #nn.Dropout(p=pDropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # shape: (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),             # shape: (batch_size, 32, 7, 7)
            #nn.Dropout(p=pDropout),
        )
        self.out = nn.Sequential(  # fully connected layer, output 10 classes
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

# choose CNN-classifier-model:
trained_Classifier = "trained models/CNN-Classifier_MNIST.pth"  # test accuracy: 99.11%
#trained_Classifier = "trained models/CNN-Classifier_FashionMNIST.pth"  # test accuracy: 89.96%
C = torch.load(trained_Classifier)

# load cGAN
# Hyperparameters:
label_knots = 64  # number of knots the label gets mapped to in both G and D
G_in_eval = False  # if G is set to eval mode during inference
uniform_input = False  # if the random input is sampled from a uniform distribution, if True: normal distribution
G_noise = 0  # add noise to G layers, has to be commented in in the rest of the code

# fixed Hyperparameters
batch_size = 100
pDropout = 0.5  # Dropout percentage
scale = 0.2  # scale for leaky ReLU
pic_knots = 512 - label_knots

# define the class for the Generator
class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.fc1_1 = nn.Linear(100, pic_knots)
        self.fc1_1_bn = nn.BatchNorm1d(pic_knots)
        self.fc1_2 = nn.Linear(10, label_knots)
        self.fc1_2_bn = nn.BatchNorm1d(label_knots)
        self.dropout1 = nn.Dropout(p=pDropout)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=pDropout)
        self.fc3 = nn.Linear(512, 784)
        self.fc3_bn = nn.BatchNorm1d(784)

    def forward(self, input, label):
        x1 = F.leaky_relu(self.fc1_1_bn(self.fc1_1(input)), negative_slope=scale)
        x2 = F.leaky_relu(self.fc1_2_bn(self.fc1_2(label)), negative_slope=scale)
        x = torch.cat([x1, x2], 1)
        #x = x + G_noise * torch.randn(1, 512, dtype=torch.float) # additional random noise
        x = self.dropout1(x)
        #x = F.dropout(x, pDropout)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), negative_slope=scale)
        #x = x + G_noise * torch.randn(1, 512, dtype=torch.float)  # additional random noise
        x = self.dropout2(x)
        #x = F.dropout(x, pDropout)
        output = torch.tanh(self.fc3_bn(self.fc3(x)))
        #output = torch.sigmoid(self.fc3_bn(self.fc3(x)))  # deactivate data rescaling in train and loop
        return output

# define the class for the Discriminator
class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1_1 = nn.Linear(784, pic_knots)
        #self.fc1_1_bn = nn.BatchNorm1d(pic_knots)
        self.fc1_2 = nn.Linear(10, label_knots)
        #self.fc1_2_bn = nn.BatchNorm1d(label_knots)
        self.dropout1 = nn.Dropout(p=pDropout)
        self.fc2 = nn.Linear(512, 512)
        #self.fc2_bn = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=pDropout)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, input, label):
        input = self.flatten(input)
        x1 = F.leaky_relu(self.fc1_1(input), negative_slope=scale)
        x2 = F.leaky_relu(self.fc1_2(label), negative_slope=scale)
        x = torch.cat([x1, x2], 1)
        x = self.dropout1(x)
        #x = F.dropout(x, pDropout)
        x = F.leaky_relu(self.fc2(x), negative_slope=scale)
        x = self.dropout2(x)
        #x = F.dropout(x, pDropout)
        output = torch.sigmoid(self.fc3(x))
        return output


#load a cGAN-model:

#MNIST
#trained_cGAN = "trained models/cGAN/cGAN_model6.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model7.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model8.pth"
trained_cGAN = "trained models/cGAN/cGAN_model9.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model10.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model11.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model12.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model13.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model14.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model15.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model16.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model17.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model18.pth"

#FashionMNIST
#trained_cGAN = "trained models/cGAN/cGAN_model19.pth"
#trained_cGAN = "trained models/cGAN/cGAN_model20.pth"

(G, _) = torch.load(trained_cGAN)


# to make sure that test set is always 10.000:
epochs = 10000//batch_size

#test it:
print("Testing...")
C.eval()
if G_in_eval == True:
    G.eval()
else:
    G.train()
total = 0
correct = 0
with torch.no_grad():
    for k in range(epochs):  # same size as test set MNIST
        if uniform_input == True:
            x_rand = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
        else:
            x_rand = torch.randn(batch_size, 100, dtype=torch.float)  # normal
        rand_label = torch.zeros(batch_size, 10, dtype=torch.float)
        rand = torch.randint(low=0, high=10, size=(batch_size, 1))
        for i in range(batch_size):
            rand_label[i, rand[i, 0].item()] = 1
        fake_images = torch.reshape(G(x_rand, rand_label), (batch_size, 1, 28, 28))
        guess = C(fake_images)
        classification = torch.argmax(guess.data, 1)
        total += batch_size
        for i in range(batch_size):
            if rand[i, 0].item() == classification[i].item():
                correct += 1
    accuracy = correct/total * 100
    print("Test Accuracy on fake images (GAN-test):", accuracy, "%")


#show 25 images with CNN-classification
plt.figure(0)
if uniform_input == True:
    x_rand = torch.rand(25, 100, dtype=torch.float)  # uniform
else:
    x_rand = torch.randn(25, 100, dtype=torch.float)  # normal
rand_label = torch.zeros(25, 10, dtype=torch.float)
rand = torch.randint(low=0, high=10, size=(25, 1))
for i in range(25):
    rand_label[i, rand[i, 0].item()] = 1
fake_images = torch.reshape(G(x_rand, rand_label), (25, 1, 28, 28))
classifications = C(fake_images)
(_, numbers) = torch.max(classifications, 1)
for i in range(25):
    title = str(numbers[i].item()) + "label:" + str(rand[i, 0].item())
    fake_image = torch.reshape(fake_images[i, :, :, :], (28, 28))
    plt.subplot(5, 5, i + 1)
    plt.axis('off')  # deletes axis from plots
    plt.gca().set_title(title)
    plt.imshow(fake_image.detach(), cmap='gray')  # gray_r for reversed grayscale
plt.show()


# show 100 images, 10 of each class in one row
plt.figure("Overview")
if uniform_input == True:
    x_rand = torch.rand(100, 100, dtype=torch.float)  # uniform
else:
    x_rand = torch.randn(100, 100, dtype=torch.float)  # normal
rand_label = torch.zeros(100, 10, dtype=torch.float)
for i in range(100):
    rand_label[i, i//10] = 1
fake_images = G(x_rand, rand_label)
fake_images = torch.reshape(fake_images, (100, 1, 28, 28))
classification = C(fake_images)
(_, number) = torch.max(classification, 1)
for i in range(100):
    fake_image = torch.reshape(fake_images[i, :, :, :], (28, 28))
    plt.subplot(10, 10, i + 1)
    plt.axis('off')
    title = str(number[i].item()) + "label:" + str(i//10)
    #plt.gca().set_title(title)  # add classification and label to each picture
    plt.imshow(fake_image.detach(), cmap='gray')  # gray_r for reversed grayscale
plt.show()

