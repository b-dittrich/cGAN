import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# Hyperparameters:
max_epochs = 100  # number of training epochs
softlabel = 0.15  # one sided maximal soft label
random_softlabel = True  # if the label is a random number between 1 and (1-softlabel)
label_knots = 64  # number of knots the label gets mapped to in both G and D
trainDtwice = True  # if D is trained twice per epoch
G_in_eval = False  # if G is set to eval mode during inference
uniform_input = False  # if the random input is sampled from a uniform distribution, if True: normal distribution
use_tensorboard = False  # if tensorboard should be used
save_model = False  # if after training model is saved
G_noise = 0  # add noise to G layers, has to be commented in in the rest of the code
data_noise = 0  # add noise to real and fake data before giving it to the discriminator, has to be commented in in the rest of the code

# fixed Hyperparameters:
batch_size = 100
learning_rate = 0.0002
pDropout = 0.5  # Dropout percentage
scale = 0.2  # scale for leaky ReLU
pic_knots = 512 - label_knots


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
        writer.add_text('cGAN changes', 'learning rate:0.0002, architecture:label_knots=64, normal random input, train D twice, epochs:100, G never in eval mode, softlabel: 0.15 random')


# main part
start = time.time()

# load data
training_data = datasets.MNIST(  # change to FashionMNIST or MNIST
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    # one hot encoding of labels
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter(0, torch.tensor(y), value=1))
)

test_data = datasets.MNIST(  # change to FashionMNIST or MNIST
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    # one hot encoding of labels
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter(0, torch.tensor(y), value=1))
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
        #output = torch.sigmoid(self.fc3_bn(self.fc3(x)))  # deactivate data rescaling in train and test loop
        return output


G = Gen()
G.apply(init_weights)

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
    for real_data, real_labels in dataloader:
        iter_comp = (epoch-1)*max_iter + iter

        # rescale real_data to [-1,1] because of tanh() in Generator
        real_data = 2*real_data - 1

        # make real_data a bit noisy but decay over time
        #real_data = real_data + data_noise * torch.randn(batch_size, 1, 28, 28, dtype=torch.float)

        # train Discriminator
        D.train()
        if G_in_eval == True:
            G.eval()
        else:
            G.train()
        D.zero_grad()

        # evaluate Discriminator on real data
        D_real = D(real_data, real_labels)

        # evaluate Discriminator on fake data
        if uniform_input == True:
            x_rand = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
        else:
            x_rand = torch.randn(batch_size, 100, dtype=torch.float)  # normal
        rand_labels = torch.zeros(batch_size, 10, dtype=torch.float)
        rand = torch.randint(low=0, high=10, size=(batch_size, 1))
        for i in range(batch_size):
            rand_labels[i, rand[i, 0]] = 1
        fake_data = G(x_rand, rand_labels)  # + data_noise * torch.randn(batch_size, 784, dtype=torch.float)  # additional data noise
        D_fake = D(fake_data, rand_labels)

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
            # add D_loss and D-performance to tensorboard
            writer.add_scalar("Loss/DisLoss", D_loss.item(), iter_comp)
            writer.add_scalar("D-Performance/Real Data", perf_real, iter_comp)
            writer.add_scalar("D-Performance/Fake Data", perf_fake, iter_comp)

        # train Generator
        G.train()
        D.eval()
        G.zero_grad()
        if uniform_input == True:
            x_rand = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
        else:
            x_rand = torch.randn(batch_size, 100, dtype=torch.float)  # normal
        rand_labels = torch.zeros(batch_size, 10, dtype=torch.float)
        rand = torch.randint(low=0, high=10, size=(batch_size, 1))
        for i in range(batch_size):
            rand_labels[i, rand[i, 0]] = 1
        fake_data = G(x_rand, rand_labels)
        D_fake = D(fake_data, rand_labels)
        G_loss = GenLoss(D_fake)

        # update Generator
        G_loss.backward()
        G_optimizer.step()

        if use_tensorboard == True:
            # add G_loss to tensorboard
            writer.add_scalar("Loss/GenLoss", G_loss.item(), iter_comp)

        # train Discriminator again
        if trainDtwice == True:
            D.train()
            if G_in_eval == True:
                G.eval()
            else:
                G.train()
            D.zero_grad()

            # evaluate Discriminator on real data
            D_real = D(real_data, real_labels)

            # evaluate Discriminator on fake data
            if uniform_input == True:
                x_rand = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
            else:
                x_rand = torch.randn(batch_size, 100, dtype=torch.float)  # normal
            rand_labels = torch.zeros(batch_size, 10, dtype=torch.float)
            rand = torch.randint(low=0, high=10, size=(batch_size, 1))
            for i in range(batch_size):
                rand_labels[i, rand[i, 0]] = 1
            fake_data = G(x_rand, rand_labels)  # + data_noise * torch.randn(batch_size, 784, dtype=torch.float)  # additional data noise
            D_fake = D(fake_data, rand_labels)
            D_loss = DisLoss(D_real, D_fake)

            # update Discriminator
            D_loss.backward()
            D_optimizer.step()

        if iter % 10 == 0:
            print("-", end="")
        iter += 1


def TestLoop(dataloader, D, G, epoch):
    D.eval()
    if G_in_eval == True:
        G.eval()
    else:
        G.train()
    iter = 1
    size = len(dataloader.dataset)
    D_test_loss, G_test_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)  # simply zero but loss comes at a tensor
    k = 0
    with torch.no_grad():
        for real_data, real_labels in dataloader:
            real_data = 2 * real_data - 1
            D_real = D(real_data, real_labels)
            if uniform_input == True:
                x_rand = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
            else:
                x_rand = torch.randn(batch_size, 100, dtype=torch.float)  # normal
            rand_labels = torch.zeros(batch_size, 10, dtype=torch.float)
            rand = torch.randint(low=0, high=10, size=(batch_size, 1))
            for i in range(batch_size):
                rand_labels[i, rand[i, 0]] = 1
            fake_data = G(x_rand, rand_labels)
            D_fake = D(fake_data, rand_labels)

            D_test_loss += DisLoss(D_real, D_fake)
            G_test_loss += GenLoss(D_fake)

            # show one generated image per epoch
            plt.figure("Training-History")
            titel = "epoch " + str(epoch)
            if k == 0:
                pic = fake_data[0, :]
                pic = torch.reshape(pic, (28, 28))
                plt.subplot(10, max_epochs//10, epoch)
                plt.axis('off')
                plt.imshow(pic, cmap='gray')
                if use_tensorboard == True:
                    # add pic to tensorboard
                    writer.add_image(titel, pic, dataformats='HW')
                k = 1
            iter += 1

        D_test_loss /= size
        G_test_loss /= size

        if use_tensorboard == True:
            iter_comp = (epoch-1) * size + iter
            writer.add_scalar("Loss/GenLoss_Test", G_test_loss.item(), iter_comp)
            writer.add_scalar("Loss/DisLoss_Test", D_test_loss.item(), iter_comp)

        print("DisLoss =", D_test_loss.item(), ", GenLoss =", G_test_loss.item())

# Actually train the GAN
print("Start Training of cGAN")
for epoch in range(1, max_epochs + 1):
    start_time = time.time()
    print("Epoch", epoch, ":")
    #decay data_noise
    #if epoch < max_epochs/2:
    #    data_noise -= data_noise/(max_epochs/2)
    #else:
    #    data_noise = 0
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
print("Finished Training of cGAN")

end = time.time()
time_complete = end - start
print("Complete Training took", time_complete)

if save_model == True:
    # save model
    GAN = (G, D)
    filename = "cGAN_trained_"+date+daytime+".pth"  # name for saving model
    torch.save(GAN, filename)

plt.show()
# make 25 random images label
D.eval()
if G_in_eval == True:
    G.eval()
else:
    G.train()
with torch.no_grad():
    plt.figure(1)
    if uniform_input == True:
        rand_input = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
    else:
        rand_input = torch.randn(batch_size, 100, dtype=torch.float)  # normal
    rand_label = torch.zeros(batch_size, 10, dtype=torch.float)
    rand = torch.randint(low=0, high=10, size=(batch_size, 1))
    for i in range(batch_size):
        rand_label[i, rand[i, 0]] = 1
    pics = G(rand_input, rand_label)
    for i in range(25):
        title = str(rand[i, 0].item())
        image = torch.reshape(pics[i, :], (28, 28))
        plt.subplot(5, 5, i + 1)
        plt.axis('off')  # deletes axis from plots
        plt.gca().set_title(title)
        plt.imshow(image, cmap='gray')  # gray_r for reversed grayscale

plt.show()
