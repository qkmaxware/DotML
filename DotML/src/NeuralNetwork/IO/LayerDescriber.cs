namespace DotML.Network;

/// <summary>
/// Produce a human readable description of a layer
/// </summary>
public class LayerDescriber : ILayerVisitor<string> {
    public string Visit(ConvolutionLayer layer) {
        return $"{layer.FilterCount} filters of size {layer.Filters.FirstOrDefault()?.FirstOrDefault().Shape}";
    }

    public string Visit(DepthwiseConvolutionLayer layer) {
        return $"{layer.Filters.Count} kernels of size {layer.Filters?.FirstOrDefault()?.FirstOrDefault().Shape}";
    }

    public string Visit(PoolingLayer layer) {
        return $"Pooling with a size of {layer.FilterHeight}x{layer.FilterWidth}";
    }

    public string Visit(FlatteningLayer layer) {
        return $"Flatten multi-dimensional input to 1D";
    }

    public string Visit(DropoutLayer layer) {
        return $"Dropout with probability {layer.DropoutRate} to reduce overfitting";
    }

    public string Visit(LayerNorm layer) {
        return $"Normalize the inputs across the layer";
    }

    public string Visit(BatchNorm layer) {
        return $"Normalize the inputs across the entire batch";
    }

    public string Visit(DenseLinearLayer layer) {
        return $"Fully connected layer of {layer.NeuronCount} neurons";
    }

    public string Visit(ActivationLayer layer) {
        return $"Apply {layer.ActivationFunction?.GetType().Name} activation function to inputs";
    }

    public string Visit(SoftmaxLayer layer) {
        return $"Convert output to probability distribution over {layer.OutputShape.Count} classes";
    }

    public string Visit(InputCapture capture) {
        return  $"Cache the inputs of this layer to reference at a later time";
    }
}