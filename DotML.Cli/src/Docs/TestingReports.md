# Training Reports
This document describes the various reporting files generated from a **Netflow** testing session.

- [Training Reports](#training-reports)
  - [Generated Files](#generated-files)
  - [File Details](#file-details)
    - [Command.sh](#commandsh)
    - [Model.xml](#modelxml)
    - [Summary.csv](#summarycsv)
    - [Details.csv](#detailscsv)

## Generated Files
| Filename                 | Summary                                                                       |
|--------------------------|-------------------------------------------------------------------------------|
| command.sh               | The command used to initiate the testing session                              |
| model.*.xml              | The model being tested's metadata                                             |
| summary.csv              | A summary of the performance of the model against the testing dataset         |
| details.csv              | A test by test breakdown of the model's performance against the testing dataset|

## File Details
### Command.sh
The `command.sh` file contains the entire CLI command used to initiate testing. This can be used as a way to reference what arguments you provided previously. Such a reference can be useful for copying arguments from one testing session to another or using all the same arguments except one or 2 that need to be changed. 

The contents of this file may look something like this:
```sh
"netflow.dll" test %MODEL_NAME% --data-test my_data.json
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

### Summary.csv 
The `summary.csv` file contains a one line summary of the entire testing process across the entire dataset.

The contents of this file may look something like this:
```csv
LOSS-AVERAGE, LOSS-MIN, LOSS-MAX, ACCURACY, PRECISION, RECALL, F1-SCORE, TIME-TAKEN
0.05205443785345426,0.0002402607930597645,0.08622778410427073,1,1,1,1, "00:00:00.0324911"
```

### Details.csv 
The `details.csv` file contains a detailed description of the metrics for each test in the testing dataset.

The contents of this file may look something like this:
```csv
TEST-INDEX, STATUS, LOSS
0,PASSED,0.0002402607930597645
1,PASSED,0.08622778410427073
2,PASSED,0.06777789368654887
3,PASSED,0.053971812829937675
```