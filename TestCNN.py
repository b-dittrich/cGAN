import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor


# load data
test_data = datasets.MNIST(  # change to FashionMNIST or MNIST
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# define class for CNN-classifier
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


# choose CNN-classifier-model:

# trained on real data:
trained_Classifier = "trained models/CNN-Classifier_MNIST.pth"
#trained_Classifier = "trained models/CNN-Classifier_FashionMNIST.pth"

# trained on fake data:
#MNIST
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model6.pth"
# model 7 has no CNN model for GAN-train
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model8.pth"
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model9.pth"
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model10.pth"
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model11.pth"
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model12.pth"
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model13.pth"
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model14.pth"
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model15.pth"
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model16.pth"
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model17.pth"
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model18.pth"

#FashionMNIST
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model19.pth"
#trained_Classifier = "trained models/CNN_GAN-train/CNN-classifier_model20.pth"

C = torch.load(trained_Classifier)


#test it:
C.eval()
total = 0
correct = 0
with torch.no_grad():
    for image, label in test_data:
        image = torch.reshape(image, (1, 1, 28, 28))
        guess = C(image)
        (_, classification) = torch.max(guess.data, 1)
        total += 1
        correct += (label == classification).sum().item()
    accuracy = correct/total * 100
    print("Test Accuracy:", accuracy, "%")



C.eval()
plt.figure(0)  # example pictures
for i in range(25):
    pic, label = test_data[i+170]  # index could be moved arbitrarily
    pic = torch.reshape(pic, (1, 1, 28, 28))
    classification = C(pic)
    (_, number) = torch.max(classification, 1)
    titel = str(number.item()) + "label:" + str(label)
    pic = torch.reshape(pic, (28, 28))
    plt.subplot(5, 5, i + 1)
    plt.axis('off')  # deletes axis from plots
    plt.gca().set_title(titel)
    plt.imshow(pic, cmap='gray')  # gray_r for reversed grayscale
plt.show()

plt.figure("Wrong Classification")
for pic, label in test_data:
    pic = torch.reshape(pic, (1, 1, 28, 28))
    classification = C(pic)
    (_, number) = torch.max(classification, 1)
    if label != number.item():
        pic = torch.reshape(pic, (28, 28))
        plt.axis('off')
        titel = str(number.item()) + "label:" + str(label)
        plt.gca().set_title(titel)
        plt.imshow(pic, cmap='gray')
        plt.show()
