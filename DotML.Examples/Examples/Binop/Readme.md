# Binary Operators (Binop)
A generic example which is used to show a neural network capable of learning boolean binary operators. 

## Configuration
This network can be used with any of the 3 big boolean binary operators such as AND, OR, and XOR. A full list of boolean binary operators can be found in Binop.cs line 17 (the OperationType enum). Choosing the desired operator can be done with the --config argument. This can be raw json, or a path to a json file. The contents of which needs to be an dictionary with the "op" key set.
```sh
--config '{\"op\": \"And\"}'
--config '{\"op\": \"Or\"}'
--config '{\"op\": \"Xor\"}'
```
If no config is provided as a command line argument, it defaults to XOR.

## Training
No training data is required, all training data is generated dynamically when using the `train` command. The train command generates tensors where True is any value between 0.25 and 1 and False is any value between -1 and -0.25. A total of 100 training pairs are generated, 25 per each truth table entry. 

Be sure to select the correct boolean operator as defined in the [Configuration](#configuration) section.

## Running
Running the example can be done with the run sub-command. Use `Binop` for the example name and for the input argument you can provide a json array of 2 values, one for each input. You can either use booleans like true/false for each array value or numbers where -1 is false and 1 is true.  

Be sure to select the correct boolean operator as defined in the [Configuration](#configuration) section.

```sh
DotML.Examples> dotnet run -c Release -- run Binop --input "[true, true]" --config '{\"op\": \"Xor\"}'
> [true, true]
[false]
```

### Input Format
Input takes the form of:
1. floating point json arrays with 2 values, one for each input. Values -1.0f is false and 1.0f is true
2. boolean json array with 2 values, one for each input. 

### Output Format
Output takes the form of a boolean json array with 1 value (true/false).