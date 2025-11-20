# Digit Recognition
An example that showcases limited image classification by recognizing a single digit on a greyscale image. Valid digits are 0 - 9. 

## Training
Consists of a images of size 320x32 pixels. Every 32 pixels represents a new digit in the order of: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. Background colour of the images should be black (no signal) and digits should be white. Some antialiasing is allowed. 

Add as many different examples of digits as you wish. Its easy enough to use MS Paint, Paint.Net, Gimp etc to create new digit samples. Getting friends, family, or coworkers to draw a few too can really make your training more reliable. I recommend at least 10 different images for enough written variety in the digits. 

When processed, each image is converted to a greyscale tensor (1 channel) of size 32x32. These tensors are stored a .json files in the tensors directory. Some data augmentation may be added to the dataset to make it more robust including scaling, rotating, changing brightness, or bluring (adding noise). 

## Running
Running the example can be done with the run sub-command. Simply provide the example name as `Digits` and for the input argument, provide a path to an image file. This will examine the image and return the most likely digit represented in the image. This works best when the image is roughly square and the digit is white on a black background. 

```sh
DotML.Examples> dotnet run -c Release -- run Digits --input "example.png" 
> example.png
 0:0 |                              | 0.00%
 1:1 |                              | 0.00%
 2:2 |                              | 0.00%
 3:3 |                              | 0.00%
 4:4 |                              | 0.00%
 5:5 |----------------------------- | 100.00%
 6:6 |                              | 0.00%
 7:7 |                              | 0.00%
 8:8 |                              | 0.00%
 9:9 |                              | 0.00%
```

### Input Format
Input takes the form of a path to a valid image file.

### Output Format
Output takes the form of a chart indicating the probability of each digit. In the above example we can see that the image provided has a 100% change to be the digit 5. 