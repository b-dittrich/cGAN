import os
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


use_tensorboard = False  # if tensorboard should be used
save_model = False  # if after training model is saved

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
        writer.add_text('CNN Changes', 'batch_size:100, epochs:10, learning_rate:0.001, pDropout:0.25, cGAN model: 9')

# Hyperparameters for training CNN-classifier:
batch_size = 100
max_epochs = 10
learning_rate = 0.001
pDropoutCNN = 0.25  # Dropout percentage for CNN

# Hyperparameters for cGAN:
label_knots = 64  # number of knots the label gets mapped to in both G and D
G_in_eval = False  # if G is set to eval mode during inference
uniform_input = False  # if the random input is sampled from a uniform distribution, if True: normal distribution
G_noise = 0  # add noise to G layers, has to be commented in in the rest of the code

# fixed Hyperparameters for cGAN
pDropout = 0.5  # Dropout percentage of cGAN-model
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


# Weight Initialization for CNN-classifier
def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.kaiming_normal_(layer.weight)  # Kaiming Initialisiation

# define class for CNN-classifier
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(      # input shape: (batch_size, 1, 28, 28)
            nn.Conv2d(1, 16, 5, 1, 2),   # shape: (batch_size, 16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),             # shape: (batch_size, 16, 14, 14)
            nn.Dropout(p=pDropoutCNN),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # shape: (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),             # shape: (batch_size, 32, 7, 7)
            nn.Dropout(p=pDropoutCNN),
        )        # fully connected layer, output 10 classes
        self.out = nn.Sequential(
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

C = CNN()
C.apply(init_weights)


optimizer = torch.optim.Adam(C.parameters(), lr=learning_rate)

Loss = nn.CrossEntropyLoss()


def TrainLoop(C, epoch):
    iter = 0
    running_loss = 0
    for k in range(500):  # same size as train set in MNIST
        if uniform_input == True:
            rand_inputs = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
        else:
            rand_inputs = torch.randn(batch_size, 100, dtype=torch.float)  # normal
        rand_labels = torch.zeros(batch_size, 10, dtype=torch.float)
        rand = torch.randint(low=0, high=10, size=(batch_size, 1))
        for i in range(batch_size):
            rand_labels[i, rand[i, 0]] = 1
        fake_images = G(rand_inputs, rand_labels)
        C.zero_grad()
        rand = torch.reshape(rand, [batch_size])
        fake_images = torch.reshape(fake_images, (batch_size, 1, 28, 28))
        perc = C(fake_images)
        loss = Loss(perc, rand)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if use_tensorboard == True:
            writer.add_scalar("Loss/CLoss", loss.item(), 500*(epoch-1)+iter)  # add loss to tensorboard
        if iter % 50 == 0:
            print("-", end="")
        iter += 1
    print("Loss:", running_loss)

def TestLoop(C):
    total = 0
    correct = 0
    with torch.no_grad():
        for k in range(100):  # same size as valid set in MNIST
            if uniform_input == True:
                rand_inputs = torch.rand(batch_size, 100, dtype=torch.float)  # uniform
            else:
                rand_inputs = torch.randn(batch_size, 100, dtype=torch.float)  # normal
            rand_labels = torch.zeros(batch_size, 10, dtype=torch.float)
            rand = torch.randint(low=0, high=10, size=(batch_size, 1))
            for i in range(batch_size):
                rand_labels[i, rand[i, 0]] = 1
            fake_images = G(rand_inputs, rand_labels)
            fake_images = torch.reshape(fake_images, (batch_size, 1, 28, 28))
            guesses = C(fake_images)
            (_, classification) = torch.max(guesses.data, 1)
            rand = torch.reshape(rand, [batch_size])
            total += batch_size
            correct += (rand == classification).sum().item()
        accuracy = correct/total * 100
        print("Validation Accuracy:", accuracy, "%")


# put G in eval or train according to how it was trained
if G_in_eval == True:
    G.eval()
else:
    G.train()
print("Start Training of Classifier")
for epoch in range(1, max_epochs + 1):
    print("Epoch", epoch, ":")
    C.train()
    TrainLoop(C, epoch)
    C.eval()
    TestLoop(C)
print("Finished Training of Classifier")

if save_model == True:
    filename = "CNN-Classifier_trained_"+date+daytime+"fake.pth"  # name for saving model
    torch.save(C, filename)


# plot 25 images with classification
C.eval()
plt.figure(0)
if uniform_input == True:
    x_rand = torch.rand(25, 100, dtype=torch.float)  # uniform
else:
    x_rand = torch.randn(25, 100, dtype=torch.float)  # normal
rand_label = torch.zeros(25, 10, dtype=torch.float)
rand = torch.randint(low=0, high=10, size=(25, 1))
for i in range(25):
    rand_label[i, rand[i, 0].item()] = 1
fake_images = G(x_rand, rand_label)
fake_images = torch.reshape(fake_images, (25, 1, 28, 28))
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

