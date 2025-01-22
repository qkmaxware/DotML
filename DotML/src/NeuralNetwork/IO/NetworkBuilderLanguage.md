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
```dockerfile
FROM SCRATCH

ADD Convolution stride=1 filters=5 kernels=6 size=3
ADD MaxPadding size=2 stride=2
ADD Dense neurons=50
ADD Softmax 
```

### Building a network from an existing template
```dockerfile
FROM AlexNet:v5

NAME "My Custom AlexNet"

REMOVE $2
MODIFY $1 5
```