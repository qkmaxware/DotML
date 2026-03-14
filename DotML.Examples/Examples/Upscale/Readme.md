# Upscaling
An example that showcases image upscaling using Luminance channel only. 

## Training
Consists of a images of size any size pixels. During pre-processing the images are cut into tiles of size 17 * UPSCALING_FACTOR. These tiles are then used for training the upscaling network.

The training dateset needs to be downloaded from the internet and is not provided by this repo.

1. Scan through Google Images, or take your own pictures of your own
   1. Images should be high quality, preprocessing will create low-res versions automatically

Use the `train` subcommand with `--raws` argument to process the images into trainable tensors.

When processed, each image is converted to a series of greyscale tensors (1 channel) of size 17x17. These tensors are stored in two binary files in the tensors subdirectory which will be the training data source during the training routine. 

## Running
Running the example can be done with the `run` sub-command. Simply provide the example name as `Upscale` and for the input argument, provide a path to an image file. This will take the image, convert it to a luminance greyscale image and then upscale the image. 

### Input Format
Input takes the form of a path to a valid image file.

### Output Format
Output takes the form of a greyscale image that should be UPSCALING_FACTOR times larger than the original. There is no colour in this example but colour could be trivially added using bilinear interpolation from the original image. 