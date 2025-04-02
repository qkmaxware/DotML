# Training Reports
This document describes the various reporting files generated from a **Netflow** training session.

- [Training Reports](#training-reports)
  - [Generated Files](#generated-files)
  - [File Details](#file-details)
    - [Command.sh](#commandsh)
    - [Weights Directory](#weights-directory)
    - [Trainer-Config.yaml](#trainer-configyaml)
    - [Model.xml](#modelxml)
    - [Network-Description.md](#network-descriptionmd)
    - [Network-Diagram.md](#network-diagrammd)
    - [Performance.csv](#performancecsv)
    - [Testing.csv](#testingcsv)
    - [Validation.csv](#validationcsv)


## Generated Files
| Filename                 | Summary                                                                       |
|--------------------------|-------------------------------------------------------------------------------|
| **weights/**             | Directory with all intermediate weights as determined by the retention policy |
| command.sh               | The command used to initiate the training session                             |
| model.*.xml              | The model being trained's metadata                                            |
| network-description.md   | A description of the network and it's layers                                  |
| network-diagram.svg      | A visual description of the network and it's layers                           |
| performance.csv          | Runtime metrics for the training process                                      |
| readme.md                | A readme file with information about the various training reports captured    |
| testing.csv              | The model's metrics per training epoch against the testing dataset            |
| trainer-config.yaml      | The configuration options used for the training procedure                     |
| validation.csv           | The model's metrics per training epoch against the validation dataset         |

## File Details
### Command.sh
The `command.sh` file contains the entire CLI command used to initiate training. This can be used as a way to reference what arguments you provided previously. Such a reference can be useful for copying arguments from one training session to another or using all the same arguments except one or 2 that need to be changed. 

The contents of this file may look something like this:
```sh
"netflow.dll" fit %MODEL_NAME% --data-training my_data.json --learning-rate 0.1 --patience 2 --accuracy 0.1
```
 
### Weights Directory
The `weights\` directory stored all intermediate weights for the network during training. Weights have the potential to be stored for each epoch of training. As such this directory can grow to be quite large. Weights will always be stored in the safetensors file format.

Which epochs have their weights saved is dependent upon the retention policy provided as a cli-argument. The default policy, if none is provided as an argument, is to not save any intermediate weights. Some of the provided retention policies include:
| Policy Name      | Saved Weights after Epoch |
|------------------|---------------------------|
| none             | Save no weights           |
| all              | Save weights after each epoch | 
| most_recent      | Save the weights for the last run epoch | 
| last5            | Save the last 5 epochs of weights |
| last10           | Save the last 10 epochs of weights |
| smallest_loss    | Save only the epoch with the smallest average loss against all pairs in the validation data |
| highest_accuracy | Save only the epoch with the highest accuracy against all pairs in the validation data |
| most_passed      | Save only the epoch with the highest number of passed tests in the validation data |


The contents of this directory may look something like this:
```
- weights/
    - epoch-1.safetensors
    - epoch-2.safetensors
    - ...
    - epoch-50.safetensors
```

### Trainer-Config.yaml
This file contains all the configuration options for the network trainer. This includes values as determined by the command line arguments as well as values determined automatically by Netflow. These configuration options are stored in [YAML](https://en.wikipedia.org/wiki/YAML) format. This can be useful for determining exactly what behaviours the training routine will perform. 

The contents of this file may look something like this:
```yaml
Trainer:
    Type: EnumerableBatchTrainer`1
    Epochs: 500
    LearningRate: 0.1
    LearningRateOptimizer: DotML.Network.Training.AdamOptimizer
    EnableGradientClipping: False
    ClippingThreshold: n/a
    ClippingThresholdSynapses: 10
    ClippingThresholdBiases: 5
    EarlyStop: True
    EarlyStopAccuracy: 0.1
    EarlyStopPatience: 2
    LossFunction: MeanSquaredError
    ValidationReport: DotML.Network.Training.DefaultValidationReport
    Profiler: DotML.Network.Training.DefaultProfilingReport
    Regularization: DotML.Network.Training.NoRegularization
    NetworkInitializer: DotML.Network.Initialization.NormalXavierInitialization
    BatchSize: 8
```

### Model.xml
The `model.*.xml` file is used to store the existing metadata for the network being trained. The `*` in the filename is the GUID of the network architecture. This is most useful when determining which network a particular set of reports apply to if some time has passed between training the network and reviewing the training reports. 

The contents of this file may look something like this:
```xml
<?xml version="1.0" encoding="utf-16"?>
<ModelInfo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <ClassLabels />
  <Tags>
    <string>xor</string>
  </Tags>
  <TrainingMetadata>
    <TrainingDuration>0:00:00:00.8820128</TrainingDuration>
    <Accuracy>1</Accuracy>
    <Precision>1</Precision>
    <Recall>1</Recall>
    <MinLoss>0.0002402607930597645</MinLoss>
    <MaxLoss>0.08622778410427073</MaxLoss>
    <AvgLoss>0.05205443785345426</AvgLoss>
  </TrainingMetadata>
</ModelInfo>
```

### Network-Description.md
The `network-description.md` file stores a written description of the network being trained. This is not guaranteed to be the same for all types of network and may be subject to change. At a minimum this file will describe each of the layers in the network. The purpose of this file is to provide enough information for others to be able to replicate the network architecture from scratch regardless of the AI framework being employed. 

### Network-Diagram.md
The `network-diagram.md` files stores a visual description of the network being trained. This is mainly for creating visualizations to help you describe your architecture to others. 

### Performance.csv
The `performance.csv` file is a comma-separated-values spreadsheet which details how much time the trainer has spent in each component of the training routine. This is most useful for developer debugging in order to determine performance bottlenecks for the trainer being used. 

The contents of this file may look something like this:
```csv
Benchmark-Name, Time-Min (s), Time-Max (s), Time-Average (s), Time-Total (s), Sample-Count
Backpropagation, 6.63E-05, 0.0064227, 0.0001571, 0.0306382, 195
Feed Forward, 4.13E-05, 0.0002571, 7.71E-05, 0.0150401, 195
Weight Update, 5.3E-06, 0.0016558, 1.78E-05, 0.0034792, 195
```

### Testing.csv
The `testing.csv` file is a comma-separated-values spreadsheet which details the performance metrics of the network after each epoch of training against the testing data-set. 

Metrics that may be tracked could include:
| Metric       | Description |
|--------------|-------------|
| Tests-Passed | Test cases where the loss is less than the desired accuracy |
| Tests-Failed | Test cases where the loss is greater than the desired accuracy |
| Loss-Average | The average loss across all test cases |
| Loss-Max     | The maximum loss across all test cases |
| Loss-Min     | The minimum loss across all test cases |
| Accuracy     | The accuracy of the network across all test cases (assuming output is a probability) |
| Precision    | The precision of the network across all test cases (assuming output is a probability) |
| Recall       | The accuracy of the network across all test cases (assuming output is a probability) |
| F1-Score     | The combined f1-score of the network across all test cases (assuming output is a probability) |
| Time-Taken   | The total amount of time taken for the training epoch |

### Validation.csv
The `validation.csv` file is a comma-separated-values spreadsheet which details the performance metrics of the network after each epoch of training against the validation data-set. 

Metrics that may be tracked could include:
| Metric | Description |
|--------------|-------------|
| Tests-Passed | Test cases where the loss is less than the desired accuracy |
| Tests-Failed | Test cases where the loss is greater than the desired accuracy |
| Loss-Average | The average loss across all test cases |
| Loss-Max     | The maximum loss across all test cases |
| Loss-Min     | The minimum loss across all test cases |
| Accuracy     | The accuracy of the network across all test cases (assuming output is a probability) |
| Precision    | The precision of the network across all test cases (assuming output is a probability) |
| Recall       | The accuracy of the network across all test cases (assuming output is a probability) |
| F1-Score     | The combined f1-score of the network across all test cases (assuming output is a probability) |
| Time-Taken   | The total amount of time taken for the training epoch |