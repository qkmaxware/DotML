# Tensor Extra Operations
This folder contains extra tensor operations that are not able to be implemented against INumber values, but are still useful for extremely common value types (like float and double).

eg. Floor and Ceiling can't be implemented for all INumbers but can be implemented for all IFloatingPoint numbers. This is because those operations make no sense for other numeric types like int. However since float and double are so popular for use in tensors these can be defined for those common cases. 