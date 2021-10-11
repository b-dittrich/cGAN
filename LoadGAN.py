import torch
import matplotlib.pyplot as plt
from torch import nn

# Hyperparameters:
G_in_eval = True  # if G is set to eval mode during inference
uniform_input = False  # if the random input is sampled from a uniform distribution, if True: normal distribution

# fixed Hyperparameters:
pDropout = 0.5  # Dropout percentage
scale = 0.2  # scale for leaky ReLU

# define the class for the Generator
class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.architecture = nn.Sequential(
            nn.Linear(100, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(scale),
            nn.Dropout(p=pDropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(scale),
            nn.Dropout(p=pDropout),
            nn.Linear(512, 784),
            nn.BatchNorm1d(784),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.architecture(input)
        return output

# define the class for the Discriminator
class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.flatten = nn.Flatten()
        self.architecture = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(scale),
            nn.Dropout(p=pDropout),
            nn.Linear(512, 512),
            nn.LeakyReLU(scale),
            nn.Dropout(p=pDropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.flatten(input)
        output = self.architecture(input)
        return output

# choose GAN-model
#trained_GAN = "trained models/GAN/GAN_model1.pth"
#trained_GAN = "trained models/GAN/GAN_model2.pth"
#trained_GAN = "trained models/GAN/GAN_model3.pth"
trained_GAN = "trained models/GAN/GAN_model4.pth"
#trained_GAN = "trained models/GAN/GAN_model5.pth"

(G, D) = torch.load(trained_GAN)


# let Generator generate 25 images
if G_in_eval == True:
    G.eval()
else:
    G.train()
D.eval()
with torch.no_grad():
    plt.figure("Fake Images")
    if uniform_input == True:
        rand_input = torch.rand(100, 100, dtype=torch.float)  # uniform
    else:
        rand_input = torch.randn(100, 100, dtype=torch.float)  # normal
    pics = G(rand_input)
    for i in range(25):
        pic = pics[i, :]
        pic = torch.reshape(pic, (28, 28))
        plt.subplot(5, 5, i + 1)
        plt.axis('off')  # deletes axis from plots
        plt.imshow(pic, cmap='gray')  # gray_r for reversed grayscale
plt.show()
