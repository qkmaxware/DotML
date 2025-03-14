# DotML NetFlow
A command line utility for building, managing, training, and running local neural networks using the DotML library (same repo).
- [DotML NetFlow](#dotml-netflow)
  - [Install](#install)
  - [Uninstall](#uninstall)
  - [Update](#update)
  - [Usage](#usage)
    - [Building a Neural Network](#building-a-neural-network)
      - [Example](#example)
    - [Listing Neural Networks](#listing-neural-networks)
      - [Example](#example-1)
    - [Deleting Neural Networks](#deleting-neural-networks)
      - [Example](#example-2)
    - [Modifying a Neural Network](#modifying-a-neural-network)
      - [Example - Adding tags](#example---adding-tags)
      - [Example - Removing tags](#example---removing-tags)
      - [Example - Importing weights](#example---importing-weights)
      - [Example - Exporting weights](#example---exporting-weights)
      - [Example - Transfering weights](#example---transfering-weights)
    - [Training a Neural Network](#training-a-neural-network)
      - [Example](#example-3)
    - [Running a Trained Network](#running-a-trained-network)
      - [Example - From STDIN](#example---from-stdin)
      - [Example - From File](#example---from-file)
  - [Saved Data](#saved-data)
    - [NETFLOW\_HOME/Models](#netflow_homemodels)
    - [NETFLOW\_HOME/Training](#netflow_hometraining)


## Install
1. Clone or Download repo
2. Using dotnet 8 or newer run the following commands
    - dotnet pack DotML.Cli.csproj -c Release
    - dotnet tool install --configfile none.nuget.config --global --add-source ./nupkg netflow
  
## Uninstall
1. Using dotnet cli run the following commands
    - dotnet tool uninstall --global netflow

## Update 
1. Just combine Install and uninstall such that
    - dotnet tool uninstall --global netflow; dotnet pack DotML.Cli.csproj -c Release; dotnet tool install --configfile none.nuget.config --global --add-source ./nupkg netflow
  

## Usage
### Building a Neural Network
The "build" command can be used to build a neural network from a "netbuild" file and store that in netflow's data directory for later training and running. Netbuild scripts are instructions indiating how to build a sequential neural network from layers. Netbuild is designed to look similar to other build scripts like dockerfiles. For a full reference of the netbuild syntax, see [DotML/src/NeuralNetwork/IO/Netlang.md](../DotML/src/NeuralNetwork/IO/Netlang.md).

In reality nothing is really "built" from using this command. What it does is verify that the netbuild script creates a valid network architecture and if so it gives the architecture a globally unique identifier (guid) and copies the build script into the data directory along with a metadata file for information about the created network. 

#### Example
// file xor.netbuild
```dockerfile
# Create a network from scratch with an input size of 1 channel, 2 rows, 1 column 
FROM SCRATCH INPUT 1 2 1 

# Create a 2->2->1 network to solve the xor problem. 
# The first 2 is handled by the input "layer".
ADD dense neurons=2
ADD activation fn=tanh
ADD dense neurons=1
ADD activation fn=tanh
```

// Shell/Terminal command
```sh
> netflow build xor.netbuild --tag xor
```

### Listing Neural Networks
To list all built networks you can use the "list" command. Each network is listed by their tags in the name column and the network guid is referenced in the architecture column. If an architecture has no tags, it it's name is just it's guid. If an architecture has multiple tags then it will be listed multiple times, once for each tag.

#### Example
// Shell/Terminal command
```sh
> netflow list
```

### Deleting Neural Networks
The delete an existing built network you can use the "rm" command. Specify the architecture name with either one of it's tags, or its full guid. Note that when looking for a network the tag name need not match completely as it is just a contains operation. If using a guid however the full guid must match. 
#### Example
// Shell/Terminal command
```sh
> netflow rm xor
```

### Modifying a Neural Network
The "mod" command can be use to modify an existing network. It can be used to add or remove tags, transfer weights, and more.

#### Example - Adding tags
// Shell/Terminal command
```sh
# Add a tag to the xor network
> netflow mod xor tags --add "new-tag"
```

#### Example - Removing tags
// Shell/Terminal command
```sh
# Remove a tag from the xor network
> netflow mod xor tags --rm "old-tag"
```

#### Example - Importing weights
// Shell/Terminal command
```sh
# Transfer pre-trained weights to the xor network
> netflow mod xor weights --import "pretrained.safetensors"
```

#### Example - Exporting weights
// Shell/Terminal command
```sh
# Transfer trained weights from the xor network to file
> netflow mod xor weights --export ".\trained.safetensors"
```

#### Example - Transfering weights
// Shell/Terminal command
```sh
# Transfer trained weights from the xor network to the xor2 network
> netflow mod xor weights --transfer xor2
```

### Training a Neural Network
Training a neural network is done using the "fit" command. The command does have some "smart" descision making in terms of chosing the correct training options. However, this may not always be accurate and as such this command can be customized with many arguments to tailor the training to your specific use case. 

At a minimum training requires that a network tag/guid be provided as well as a training data-set. The training data set can be in many possible forms including JSON, categorized CSV, serialized DotML Training Set and more. You can also provide additional data sets for validation and testing as well. 

#### Example
In this example we are going to train the xor network using a training set in the form of a JSON file. 

// file training.json
```json
[
    { "Input": [0, 0], "Output": [0] },
    { "Input": [0, 1], "Output": [1] },
    { "Input": [1, 0], "Output": [1] },
    { "Input": [1, 1], "Output": [0] },
]
```

// Shell/Terminal command
```sh
> netflow fit xor --data-training "training.json"
```

### Running a Trained Network
Once a network has been trained you probably want to use it for some tasks. Running a network can be handled by using the "run" command. For this command you are required to specify 3 bits of information. First, the tag or guid of the network to run, then the input data to send to the network, and the specification of the encoding of the input (the embedding). 

Input can be provided via the input command line argument, or if that is missing, the text from standard input. 

There are numerous different embeddings which are used to convert the inputs to neural network input tensors such as JSON, CSV, IMAGE, etc. Additionally there are numeroud different decoders which convert the output tensors from a network back into something human understandable like a probability chart or an output image. 

#### Example - From STDIN
In this example we are going to run the xor network against a test vector by passing the input in via a standard input pipe. The embedding argument indicates that the input should be treated as a vector which is encoded as a json file. 

// Shell/Terminal command
```sh
> echo "[0, 1]" | netflow run xor --embedding json
```

#### Example - From File
In this example we are doing the same thing as above, but we are using a file to store our data. 

// file input.json
```json
[0, 1]
```

// Shell/Terminal command
```sh
netflow run xor --embedding json -f "input.json"
```

## Saved Data
All data is stored in the local application data directory (`%LocalAppData%/DotML.NetFlow` on windows). You may specify a different directory to use for app storage using the `NETFLOW_HOME` environment variable.

### NETFLOW_HOME/Models
This folder contains all the network architectures as *.netbuild instructions as well as their metadata as a *.xml file and their trained weights as a *.safetensors file. 

This folder is completely managed by the commands above such as "build", "rm", and "mod" and does not need to be managed by the user. 

### NETFLOW_HOME/Training
This folder contains reports and statistics for different training sessions. Each training session is it's own folder, named with the date/time of the start of the training. CSV reports, network diagrams, intermediate weights, and more are stored here. At the end of any training session using the "fit" command, the directory containing it's reports is displayed to the user. 

The most common use of this folder is to review the results of a given training session. For instance, one could be examining the loss or f1 score of a network over time or they could be looking at how the weights changed after each epoch (by default no intermediate weights are saved  However there are different retention strategies available for intermediate weights). 

This folder is **not** managed by the netflow cli tool and needs to be managed by the user. If intermediate weights are being saved during training this folder can grow to be quite large and should be cleaned up on a regular basis.   