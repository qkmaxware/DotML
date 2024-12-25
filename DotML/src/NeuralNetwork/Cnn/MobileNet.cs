namespace DotML.Network;

/// <summary>
/// Utility methods related to the MobileNet Convolutional Neural Network architecture
/// </summary>
public static class MobileNet {

    // TODO customizable activation function

    private static IEnumerable<IConvolutionalFeedforwardNetworkLayer> DepthwiseBlock(Shape3D size, int stride, int kernel_size=3, Padding padding = Padding.Same, ActivationFunction? activation = null) {
        var first = new DepthwiseConvolutionLayer(input_size: size, padding: padding, stride: stride, filter: ConvolutionFilter.Make(1, size.Channels, kernel_size)[0]);
        yield return first;
        yield return new LayerNorm(input_size: first.OutputShape);
        yield return new ActivationLayer(input_size: first.OutputShape, activation: activation ?? ReLU.Instance);
    }

    private static IEnumerable<IConvolutionalFeedforwardNetworkLayer> PointwiseBlock(Shape3D size, int filter_count, int kernel_size = 1, int stride=1, Padding padding = Padding.Same, bool dropout = false, double dropout_percent = 0.1, ActivationFunction? activation = null) {
        var first = new ConvolutionLayer(input_size: size, padding: padding, stride: stride, filters: ConvolutionFilter.Make(filter_count, size.Channels, kernel_size));
        yield return first;
        yield return new LayerNorm(input_size: first.OutputShape);
        yield return new ActivationLayer(input_size: first.OutputShape, activation: activation ?? ReLU.Instance);

        if (dropout) {
            yield return new DropoutLayer(input_size: first.OutputShape, dropoutRate: dropout_percent);
        }
    }

    /// <summary>
    /// Construct a network using the MobileNet v1 architecture
    /// </summary>
    /// <param name="output_classes">number of output classifications</param>
    /// <returns>network</returns>
    public static ConvolutionalFeedforwardNetwork MakeV1(int output_classes, ActivationFunction? activation = null) {
        const int IMG_WIDTH = 224;
        const int IMG_HEIGHT = 224;
        const int IMG_CHANNELS = 3;

        activation = activation ?? ReLU.Instance;

        return new ConvolutionalFeedforwardNetwork(
            // Input layer & first convolution
            new ConvolutionLayer(input_size: new Shape3D(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH), padding: Padding.Same, stride: 2, filters: ConvolutionFilter.Make(32, 3, 3))
            .Then((size) => new LayerNorm(input_size: size))
            .Then((size) => new ActivationLayer(input_size: size, activation: activation))
            // First depthwise separable convolution block
            .Then((size) => DepthwiseBlock(size, stride: 1, kernel_size: 3, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 64, stride: 1, activation: activation))
            // Second depthwise separable convolution block
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 2, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 128, stride: 1, activation: activation))
            // Third depthwise separable convolution block
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 1, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 128, stride: 1, activation: activation))
            // Fourth depthwise separable convolution block
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 2, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 256, stride: 1, activation: activation))
            // Fifth depthwise separable convolution block
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 1, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 256, stride: 1, activation: activation))
            // Sixth depthwise separable convolution block
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 2, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 512, stride: 1, activation: activation))
            // Seventh depthwise separable convolution block (repeated 5 times)
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 1, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 512, stride: 1, activation: activation))
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 1, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 512, stride: 1, activation: activation))
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 1, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 512, stride: 1, activation: activation))
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 1, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 512, stride: 1, activation: activation))
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 1, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 512, stride: 1, activation: activation))
            // Eighth depthwise separable convolution block
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 2, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 1024, stride: 1, activation: activation))
            // Ninth depthwise separable convolution block
            .Then((size) => DepthwiseBlock(size, kernel_size: 3, stride: 1, activation: activation))
            .Then((size) => PointwiseBlock(size, filter_count: 1024, stride: 1, activation: activation))
            // Pooling layer
            .Then((size) => new LocalAvgPoolingLayer(input_size: size, size: 7, stride: 1))
            // Output layers
            .Then((size) => new FullyConnectedLayer(input_size: size.Count, neurons: 1024))
            .Then((size) => new ActivationLayer(input_size: size, activation: activation))
            .Then((size) => new FullyConnectedLayer(input_size: size.Count, output_classes))
            .Then((size) => new SoftmaxLayer(size: size.Count))
        );
    }
}