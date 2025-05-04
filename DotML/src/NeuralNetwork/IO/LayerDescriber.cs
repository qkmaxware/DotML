namespace DotML.Network;

/// <summary>
/// Produce a human readable description of a layer
/// </summary>
public class LayerDescriber : ILayerOutputVisitor<string> {
    public string Visit(ConvolutionLayer layer) {
        return $"Convolution from {layer.Filters.FirstOrDefault()?.Count()} channels to {layer.FilterCount} channels with kernels of size {layer.Filters.FirstOrDefault()?.FirstOrDefault().Shape}";
    }

    public string Visit(DepthwiseConvolutionLayer layer) {
        return $"Depth-wise convolution of {layer.Filters.Count} kernels of size {layer.Filters?.FirstOrDefault()?.FirstOrDefault().Shape}";
    }

    public string Visit(TransposeConvolutionLayer layer) {
        return $"Transpose convolution from {layer.Filters.Count} channels to {layer.Filters.FirstOrDefault()?.Count()} channels with kernels of size {layer.Filters?.FirstOrDefault()?.FirstOrDefault().Shape}";
    }

    public string Visit(PixelShuffle layer) {
        return $"Shuffle pixels from {layer.InputShape.Channels} channels to {layer.OutputShape.Channels} channels to upscale resolution by {layer.UpscalingFactor}x.";
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
        return  $"Cache the inputs of this layer to reference at a later time ({capture.UID()})";
    }

    public string Visit(AdditionSkipConnection skip) {
        return  $"Add the input to this layer with a previous layer's output ({skip.CaptureSource.UID()})";
    }
}