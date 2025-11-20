## Training
Consists of a images of size 224x128 pixels. Every 32x32 pixel block represents a new letter. Background colour of the images should be black (no signal) and digits should be white. Some antialiasing is allowed. 

The letters are organized in the images as follows:

A B C D E F G
H I J K L M N 
O P Q R S T U
V W X Y Z ! ?

Add as many different examples of letters as you wish. Its easy enough to use MS Paint, Paint.Net, Gimp etc to create new digit samples. Getting friends, family, or coworkers to draw a few too can really make your training more reliable. I recommend at least 10 different images for enough written variety in the digits. 

When processed, each image is converted to a greyscale tensor (1 channel) of size 32x32. These tensors are stored a .json files in the tensors directory. Some data augmentation may be added to the dataset to make it more robust including scaling, rotating, changing brightness, or bluring (adding noise). 