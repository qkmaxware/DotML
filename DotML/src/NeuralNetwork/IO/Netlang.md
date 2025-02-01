# Netlang
A dockerfile like language for serializing and constructing neural networks.

## Syntax
```bnf
script  ::= src command*
src     ::= "FROM" identifier (":" identifier)?
command ::= add | arg | name | remove | replace

add     ::= "ADD" identifier argument* ("AS" identifier)?
arg     ::= "ARG" argument*
name    ::= "NAME" identifier
remove  ::= "REMOVE" layer_reference
replace ::= "REPLACE" layer_reference "WITH" identifier argument*

argument::= identifier "=" value
layer_reference::= $\d+ | identifier

value   ::= number | identifier | ...
identifier::= \w+ | string
```

## Examples
### Building a network from scratch
The following creates a 2-2-1 network. The first line indicates that we are making a network from scratch that accepts an input of shape {channels: 1, rows: 2, columns: 1}. This is where the first 2 comes from, the input shape. It has two fully-connected or 'dense' layers consist of 2 neurons and 1 neuron respectively. This is where the next 2 and 1 come from respectively. 
```dockerfile
# Create a network from scratch to solve the XOR problem
FROM SCRATCH 
INPUT 1 2 1
NAME "XOR-221-Network"

ADD Dense neurons=2
ADD Dense neurons=1 AS output
```

### Building a network from an existing template
```dockerfile
# Create a network based on AlexNet version 5
FROM AlexNet:v5
NAME "My Custom AlexNet"

# Remove layer 2
REMOVE 2
# Replace the 1st layer with a new dense layer of 3 neurons
REPLACE 1 WITH dense neurons=3
```