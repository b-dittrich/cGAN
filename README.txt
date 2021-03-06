Codes for my report "A (conditional) Generative Adversarial Network". The PDF-file can be found here: https://www.mathematik.uni-wuerzburg.de/en/scientificcomputing/teaching/documents-lecture-notes-downloads-etc/

Instruction for the code files:

The path of the trained models must not be changed unless it is also changed in the code.
When first training a model on MNIST or FashionMNIST the dataset is automatically downloaded in a folder called 'data' in the same directory as the file.

Infos to the use of tensorboard:
	- In each file where a model is trained one can activate tensorboard (set use_tensorboard = True).
	- This creates a folder in the same directory as the python file called 'log' in which a folder gets created with the date of the run in which these information is saved as an event file.
	- This event file contains information like the Discriminator performance or Losses (see the code for more details).
	- They can be called live or after training using tensorboard.

GAN.py: trains a GAN-model on MNIST or FashionMNIST
	- The data set has to be chosen in line 57 and 64!
	- Hyperparameters can all be changed at the top (line 13-28).
	- At the end shows a picture of the training history (one generated picture per epoch) and a picture with 25 generated images.

cGAN.py: trains a cGAN-model on MNIST or FashionMNIST
	- The data set has to be chosen in line 60 and 69!
	- Hyperparameters can all be changed at the top (line 13-30), just a few extras have to be commented in in the code if one wants to use them.
	- At the end shows a picture of the training history (one generated picture per epoch) and a picture with 25 generated images with labels.

CNN-classifier.py: trains a convolutional classifier on MNIST or FashionMNIST
	- The data set has to be chosen in line 44 and 51!
	- Hyperparameters can be changed in lines 37-40 but this is not necessary to reproduce the models used in the report as both were trained with these parameters.
	- The training set is split into 50.000 for training and 10.000 for validation leaving the test set of size 10.000 for one final test at the end in TestCNN.py.
	- At the end shows a picture of 25 images with classification of the CNN-classifier and original label.

GAN-test_CNN.py: determines the GAN-test accuracy of a chosen cGAN-model with a chosen CNN-classifier-model
	- The CNN-classifier model can be chosen in line 34/35.
	- The cGAN-model can be chosen from the lines 113-129.
	- Pay attention to set the hyperparameters after line 39 as in the training of the cGAN-model!
	- At the end prints out the GAN-test accuracy and shows a picture of 25 generated images with the classification of the CNN-classifier and the original label and a picture with 100 generated images with one class per row (alternatively for each picture the classification and original label can be shown: comment in line 208).

GAN-train_CNN.py: trains a CNN-model on fake images generated by a chosen cGAN-model
	- The cGAN-model can be chosen from the lines 114-130.
	- Pay attention to set the hyperparameters after line 40 as in the training of the cGAN-model!
	- The Hyperparameters for training the CNN-classifier (lines 34-38) have not be changed to reproduce the results from the report.
	- At the end shows a picture of 25 generated images with the classification of the CNN-classifier.

TestCNN.py: tests a chosen CNN-classifier on the MNIST or FashionMNIST data set
	- The CNN-model can be chosen in the lines 44-68.
	- The data set has to be chosen in line 9!
	- At the end prints out the test respectively the GAN-train accuracy and shows a picture with 25 images with classification.
	- Additionally after that prints out images with wrong classifications as long as the program is not stopped or wrong classifications are left.

LoadGAN.py: load a trained GAN-model to print out some pictures
	- The GAN-model can be chosen in lines 57-61.
	- The hyperparameters can be changed under line 5.
	- Shows one picture with 25 generated images.

LoadcGAN.py: load a trained cGAN-model to print out some pictures
	- The cGAN-model can be chosen in lines 77-96.
	- Pay attention to set the hyperparameters after line 6 as in the training of the cGAN-model.
	- Shows one picture with 100 generated images, one row of 10 images per class, optionally with Discriminator classifications.
	- Shows another picture with 100 images, one image per class per row, where each row the Dropout percentage is reduced (ATTENTION: For this to work one needs to use F.dropout instead of nn.Dropout).
