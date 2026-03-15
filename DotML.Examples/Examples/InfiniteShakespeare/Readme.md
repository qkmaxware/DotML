# Infinite Shakespeare
An example that showcases next-token generation using a CNN network. Designed to print out an infinitely running amalgam of Shakespeare's plays. 

## Training 
The training dateset needs to be downloaded from the internet and is not provided by this repo.

1. Goto cobanov's github page and download some plays. More plays adds more variety but increases training time
   1. https://github.com/cobanov/shakespeare-dataset/tree/main/text
   2. Put them in the "raws" subdirectory
2. Use the `train` subcommand with `--raws` to convert the images to training tensors and train the network.

## Running
Running the example can be done with the `run` sub-command. Simply provide the example name as `InfiniteShakespeare` and leave the input argument blank. You will keep getting more words until you cancel the process with ctrl+c.

### Input Format
Optional, but you can provide some words to "start" the generation.

### Output Format
Output takes the form of an infinitely generating play using words and characters from all plays it was trained on. 