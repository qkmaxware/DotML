# Fruits
An example that showcases image classification for 10 different types of fruits. This example takes colour images of 128x128 pixels and determined which of the following 10 classes is represented by the image. 

1. "Apple",
2. "Avocado",
3. "Banana",
4. "Cherry",
5. "Grape",
6. "Orange",
7. "Peach",
8. "Pineapple",
9. "Strawberry",
10. "Watermelon"

## Training 
The training dateset needs to be downloaded from the internet and is not provided by this repo.

1. Scan through Google Images, or take your own pictures for each of the image classes. 
   1. Use a min of 10 images per class, ideally more
   2. Keep the number of images per class balanced
2. Use the `train` subcommand with `--raws` to convert the images to training tensors and train the network.

## Running
Running the example can be done with the `run` sub-command. Simply provide the example name as `Fruits` and for the input argument, provide a path to an image file. Images can be of any size and will be preprocessed to be 128x128 pixels as the internal network expects. Given this preprocessing, it will work better if the target is prominent in the image and the image is square.

### Input Format
Input takes the form of a path to a valid image file.

### Output Format
Output takes the form of a chart indicating the probability of each of the 10 classes.