# Cifar10
An example that showcases image classification based on the CIFAR-10 dataset. This example takes colour images of 32x32 pixels and determined which of the following 10 classes is represented by the image. 

1. "Airplane",
2. "Automobile",
3. "Bird",
4. "Cat",
5. "Deer",
6. "Dog",
7. "Frog",
8. "Horse",
9. "Ship",
10. "Truck"

## Training 
The training dateset needs to be downloaded from the internet and is not provided by this repo.

1. Download the CIFAR-10 Binary dataset (c-compatible) from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
2. Extract the .tar.gz file in the raws directory
3. Ensure you have the following files in the raws directory
   1. data_batch_1.bin
   2. data_batch_2.bin
   3. data_batch_3.bin
   4. data_batch_4.bin
   5. data_batch_5.bin
   6. test_batch.bin
4. Use the `train` subcommand with to train the network.

## Running
Running the example can be done with the `run` sub-command. Simply provide the example name as `Cifar10` and for the input argument, provide a path to an image file. Images can be of any size and will be preprocessed to be 32x32 pixels as the internal network expects. Given this preprocessing, it will work better if the target is prominent in the image and the image is square.

### Input Format
Input takes the form of a path to a valid image file.

### Output Format
Output takes the form of a chart indicating the probability of each of the 10 classes.