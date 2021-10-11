import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor


# Hyperparameters:
max_epochs = 100  # number of training epochs
softlabel = 0.15  # one sided maximal soft label
random_softlabel = True  # if the label is a random number between 1 and (1-softlabel)
trainDtwice = True  # if D is trained twice per epoch
G_in_eval = False  # if G is set to eval mode during inference
D_in_eval = True  # if D is set to eval mode during G training
uniform_input = False  # if the random input is sampled from a uniform distribution, if True: normal distribution
use_tensorboard = False  # if tensorboard should be used
save_model = False  # if after training model is saved

# fixed Hyperparameters:
batch_size = 100
learning_rate = 0.0002
pDropout = 0.5  # Dropout percentage
scale = 0.2  # scale for leaky ReLU


# some things needed for tensorboard or model saving
if (use_tensorboard == True) or (save_model == True):
    # get time/date for log data (tensorboard) or model saving
    def get_time():
        date_time = datetime.now()
        return date_time.strftime("%H%M%S")

    def get_date():
        date_time = datetime.now()
        return date_time.strftime("%Y-%m-%d")

    daytime = get_time()
    date = get_date()

    if use_tensorboard == True:
        logdir = os.path.join("log", date, daytime)
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
        # here one can write some parameters of the model which is trained to look it up later in tensorboard
        writer.add_text('GAN changes', 'Soft labels: 0.15 random, train D: twice, G,D in train during training an eval during eval, normal input')


# main part
start = time.time()

# load data
training_data = datasets.MNIST(  # change to FashionMNIST or MNIST
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(  # change to FashionMNIST or MNIST
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# Weight Initialization
def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.kaiming_normal_(layer.weight)  # Kaiming Initialisiation

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


G = Gen()
G.apply(init_weights)

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


D = Dis()
D.apply(init_weights)

# choose optimizer
G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

BCELoss = nn.BCELoss()

# define Generator Loss
def GenLoss(fake_perc):
    real_label = torch.ones(batch_size, 1, dtype=torch.float)
    loss = BCELoss(fake_perc, real_label)
    return loss

# define Discriminator Loss
def DisLoss(real_perc, fake_perc):
    if random_softlabel == True:
        real_label = torch.ones(batch_size, 1, dtype=torch.float) - softlabel * torch.rand(batch_size, 1, dtype=torch.float)  # random soft label
    else:
        real_label = (1 - softlabel) * torch.ones(batch_size, 1, dtype=torch.float)  # fixed soft label
    fake_label = torch.zeros(batch_size, 1, dtype=torch.float)
    loss = BCELoss(real_perc, real_label) + BCELoss(fake_perc, fake_label)
    return loss

# one epoch training loop
def TrainingLoop(dataloader, D, G, epoch):
    iter = 1
    max_iter = len(dataloader)
    for real_data, _ in dataloader:
        iter_comp = epoch*max_iter + iter

        # rescale real_data to [-1,1] because of tanh() in Generator
        real_data = 2*real_data - 1

        # train Discriminator
        D.train()
        if G_in_eval == True:
            G.eval()
        else:
            G.train()
        D.zero_grad()

        # evaluate Discriminator on real data
        D_real = D(real_data)

        # evaluate Discriminator on fake data
        if uniform_input == True:
            x_rand = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
        else:
            x_rand = torch.randn(batch_size, 100, dtype=torch.float)  # normal
        fake_data = G(x_rand)
        D_fake = D(fake_data)
        D_loss = DisLoss(D_real, D_fake)

        # update Discriminator
        D_loss.backward()
        D_optimizer.step()

        if use_tensorboard == True:
            # number of correctly classified real_data
            corr_real = (D_real > 0.5).sum().item()
            perf_real = corr_real / batch_size

            # number of correctly classified fake_data
            corr_fake = (D_fake < 0.5).sum().item()
            perf_fake = corr_fake / batch_size

            # add D_loss, D-performance to tensorboard
            writer.add_scalar("Loss/DisLoss", D_loss.item(), iter_comp)
            writer.add_scalar("D-Performance/Real Data", perf_real, iter_comp)
            writer.add_scalar("D-Performance/Fake Data", perf_fake, iter_comp)

        #train G
        if D_in_eval == True:
            D.eval()
        else:
            D.train()
        G.train()
        G.zero_grad()
        if uniform_input == True:
            x_rand = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
        else:
            x_rand = torch.randn(batch_size, 100, dtype=torch.float)  # normal
        fake_data = G(x_rand)
        D_fake = D(fake_data)
        G_loss = GenLoss(D_fake)

        # update Generator
        G_loss.backward()
        G_optimizer.step()

        if use_tensorboard == True:
            # add G_loss to tensorboard
            writer.add_scalar("Loss/GenLoss", G_loss.item(), iter_comp)

        #train Discriminator again
        if trainDtwice == True:
            # train Discriminator
            D.train()
            if G_in_eval == True:
                G.eval()
            else:
                G.train()
            D.zero_grad()

            # evaluate Discriminator on real data
            D_real = D(real_data)

            # evaluate Discriminator on fake data
            if uniform_input == True:
                x_rand = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
            else:
                x_rand = torch.randn(batch_size, 100, dtype=torch.float)  # normal
            fake_data = G(x_rand)
            D_fake = D(fake_data)
            D_loss = DisLoss(D_real, D_fake)

            # update Discriminator
            D_loss.backward()
            D_optimizer.step()

        if iter % 10 == 0:
            print("-", end="")
        iter += 1


def TestLoop(dataloader, D, G, epoch):
    iter = 1
    size = len(dataloader.dataset)
    D_test_loss, G_test_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)  # simply zero but loss comes at a tensor
    k = 0
    D.eval()
    if G_in_eval == True:
        G.eval()
    else:
        G.train()
    with torch.no_grad():
        for real_data, _ in dataloader:
            real_data = 2 * real_data - 1
            D_real = D(real_data)
            if uniform_input == True:
                x_rand = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
            else:
                x_rand = torch.randn(batch_size, 100, dtype=torch.float)  # normal
            fake_data = G(x_rand)
            D_fake = D(fake_data)

            D_test_loss += DisLoss(D_real, D_fake)
            G_test_loss += GenLoss(D_fake)

            # show one generated image per epoch
            plt.figure("Training-History")
            titel = "epoch " + str(epoch)
            if k == 0:
                pic = fake_data[0]
                pic = torch.reshape(pic, (28, 28))
                plt.subplot(10, max_epochs//10, epoch)
                plt.axis('off')
                plt.imshow(pic, cmap='gray')
                if use_tensorboard == True:
                    # add pic to tensorboard
                    writer.add_image(titel, pic, dataformats='HW')
                k = 1
            iter += 1

        D_test_loss /= batch_size
        G_test_loss /= batch_size

        if use_tensorboard == True:
            iter_comp = epoch * size + iter
            writer.add_scalar("Loss/GenLoss_Test", G_test_loss.item(), iter_comp)
            writer.add_scalar("Loss/DisLoss_Test", D_test_loss.item(), iter_comp)

        print("DisLoss =", D_test_loss.item(), ", GenLoss =", G_test_loss.item())


# Actually train the GAN
print("Start Training of GAN")
for epoch in range(1, max_epochs + 1):
    start_time = time.time()
    print("Epoch", epoch, ":")
    # train G and D
    G.train()
    D.train()
    TrainingLoop(train_dataloader, D, G, epoch)
    # evaluate G and D
    G.eval()
    D.eval()
    TestLoop(test_dataloader, D, G, epoch)
    end_time = time.time()
    epoch_time = end_time - start_time
    print("Time:", epoch_time)
print("Finished Training of GAN")

end = time.time()
time_complete = end - start
print("Complete Training took", time_complete)

if save_model == True:
    # save model
    GAN = (G, D)
    filename = "GAN_trained_"+date+daytime+".pth"  # name for saving model
    torch.save(GAN, filename)

plt.show()  # shows history
# make 25 random images
if G_in_eval == True:
    G.eval()
else:
    G.train()
D.eval()
with torch.no_grad():
    plt.figure(1)
    if uniform_input == True:
        rand_input = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
    else:
        rand_input = torch.randn(batch_size, 100, dtype=torch.float)  # normal
    pics = G(rand_input)
    for i in range(25):
        pic = pics[i, :]
        pic = torch.reshape(pic, (28, 28))
        plt.subplot(5, 5, i + 1)
        plt.axis('off')  # deletes axis from plots
        plt.imshow(pic, cmap='gray')  # gray_r for reversed grayscale
plt.show()
