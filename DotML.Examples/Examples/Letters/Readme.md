# Letter Recognition
An example that showcases limited image classification by recognizing a single letter on a greyscale image. Valid letters are the capital A - Z as well as ! and ?. 

## Training
Consists of a images of size 224x128 pixels. Every 32x32 pixel block represents a new letter. Background colour of the images should be black (no signal) and digits should be white. Some antialiasing is allowed. 

The letters are organized in the images as follows:

A B C D E F G
H I J K L M N 
O P Q R S T U
V W X Y Z ! ?

Add as many different examples of letters as you wish. Its easy enough to use MS Paint, Paint.Net, Gimp etc to create new digit samples. Getting friends, family, or coworkers to draw a few too can really make your training more reliable. I recommend at least 10 different images for enough written variety in the digits. 

Use the `train` subcommand with `--raws` argument to process the images into trainable tensors.

When processed, each image is converted to a greyscale tensor (1 channel) of size 32x32. These tensors are stored a .json files in the tensors directory. Some data augmentation may be added to the dataset to make it more robust including scaling, rotating, changing brightness, or bluring (adding noise). 

## Running
Running the example can be done with the `run` sub-command. Simply provide the example name as `Letters` and for the input argument, provide a path to an image file. This will examine the image and return the most likely digit represented in the image. This works best when the image is roughly square and the digit is white on a black background. 

### Input Format
Input takes the form of a path to a valid image file.

### Output Format
Output takes the form of a chart indicating the probability of each letter.