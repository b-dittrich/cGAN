import os
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor


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
        writer.add_text('CNN Changes', 'batch_size:100, epochs:10, learning_rate:0.001, pDropout:0.25, MNIST')

# Hyperparameters:
batch_size = 100
max_epochs = 10
learning_rate = 0.001
pDropout = 0.25  # Dropout percentage


# load data
training_data = datasets.MNIST(  # change to FashionMNIST or MNIST
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(  # change to FashionMNIST or MNIST
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_data, valid_data = torch.utils.data.random_split(training_data, [50000, 10000])  # split training data for extra 10k validation set
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# Weight Initialization
def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.kaiming_normal_(layer.weight)  # Kaiming Initialisiation

# define the class for the classifier
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(      # input shape: (batch_size, 1, 28, 28)
            nn.Conv2d(1, 16, 5, 1, 2),   # shape: (batch_size, 16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),             # shape: (batch_size, 16, 14, 14)
            nn.Dropout(p=pDropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # shape: (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),             # shape: (batch_size, 32, 7, 7)
            nn.Dropout(p=pDropout),
        )
        self.out = nn.Sequential(  # fully connected layer, output 10 classes
            nn.Linear(32 * 7 * 7, 10),
        )  # no softmax needed because Cross-Entropy-Loss has it inbuilt

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

C = CNN()
C.apply(init_weights)

optimizer = torch.optim.Adam(C.parameters(), lr=learning_rate)

Loss = nn.CrossEntropyLoss()

def TrainLoop(dataloader, C, epoch):
    iter = 1
    running_loss = 0
    for images, labels in dataloader:
        images = torch.reshape(images, (batch_size, 1, 28, 28))
        C.zero_grad()
        perc = C(images)
        loss = Loss(perc, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if use_tensorboard == True:
            writer.add_scalar("Loss/CLoss", loss.item(), 100*(epoch-1)+iter)  # add loss to tensorboard
        if iter % 10 == 0:
            print("-", end="")
        iter += 1
    print("Loss:", running_loss)

def TestLoop(dataloader, C):
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = torch.reshape(images, (batch_size, 1, 28, 28))
            guesses = C(images)
            (_, classification) = torch.max(guesses.data, 1)
            total += labels.size(0)
            correct += (labels == classification).sum().item()
        accuracy = correct/total * 100
        print("Validation Accuracy:", accuracy, "%")




print("Start Training of CNN-Classifier")
for epoch in range(1, max_epochs + 1):
    print("Epoch", epoch, ":")
    C.train()
    TrainLoop(train_dataloader, C, epoch)
    C.eval()
    TestLoop(valid_dataloader, C)
print("Finished Training of CNN-Classifier")

if save_model == True:
    filename = "CNN-Classifier_trained_"+date+daytime+".pth"  # name for saving model
    torch.save(C, filename)

# plot 25 images with classification
plt.figure("Test Images")
for k in range(1, 26):
    image, label = test_data[k]
    image = torch.reshape(image, (1, 1, 28, 28))
    _, guess = torch.max(C(image).data, 1)
    title = str(guess.item()) + "label:" + str(label)
    pic = image.view(28, 28)
    plt.subplot(5, 5, k)
    plt.axis('off')  # deletes axis from plots
    plt.gca().set_title(title)
    plt.imshow(pic, cmap='gray')
plt.show()

