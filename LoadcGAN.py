import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

# Hyperparameters:
label_knots = 64  # number of knots the label gets mapped to in both G and D
G_in_eval = False  # if G is set to eval mode during inference
uniform_input = False  # if the random input is sampled from a uniform distribution, if True: normal distribution
G_noise = 0  # add noise to G layers, has to be commented in in the rest of the code

# fixed Hyperparameters:
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



# choose cGAN-model:

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

(G, D) = torch.load(trained_cGAN)


# let Generator generate some images
if G_in_eval == True:
    G.eval()
else:
    G.train()
D.eval()
with torch.no_grad():
    plt.figure("Fake Images-Overview")  # 100 pictures, each row one class
    if uniform_input == True:
        rand_input = torch.rand(100, 100, dtype=torch.float)  # uniform
    else:
        rand_input = torch.randn(100, 100, dtype=torch.float)  # normal
    labels = torch.zeros(100, 10, dtype=torch.float)
    for i in range(100):
        labels[i, i//10] = 1
    pics = G(rand_input, labels)
    for i in range(100):
        pic = torch.reshape(pics[i, :], (1, 784))
        label = torch.reshape(labels[i, :], (1, 10))
        number = torch.argmax(label).item()
        classification = D(pic, label).item()
        perc = round(classification, 2) * 100
        title = "real " + str(perc) + "%, label:" + str(number)
        pic = torch.reshape(pic, (28, 28))
        plt.subplot(10, 10, i + 1)
        plt.axis('off')  # deletes axis from plots
        #plt.gca().set_title(title)  # adds discriminator label as titel to each picture
        plt.imshow(pic, cmap='gray')  # gray_r for reversed grayscale

    # look what happens when reducing dropout
    plt.figure("Reducing Dropout")
    for k in range(10):
        if uniform_input == True:
            rand_input = torch.rand(100, 100, dtype=torch.float)  # uniform
        else:
            rand_input = torch.randn(100, 100, dtype=torch.float)  # normal
        labels = torch.zeros(100, 10, dtype=torch.float)
        for i in range(100):
            labels[i, i % 10] = 1
        pics = G(rand_input, labels)
        for i in range(10):
            pic = torch.reshape(pics[i, :], (1, 784))
            label = torch.reshape(labels[i, :], (1, 10))
            number = torch.argmax(label).item()
            pic = torch.reshape(pic, (28, 28))
            plt.subplot(10, 10, 10*k + i + 1)
            plt.axis('off')  # deletes axis from plots
            plt.imshow(pic, cmap='gray')  # gray_r for reversed grayscale
        pDropout -= 0.05  # reduce dropout every row
plt.show()
