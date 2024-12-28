namespace DotML.Network;

/// <summary>
/// Utility methods related to the LeNet Convolutional Neural Network architecture
/// </summary>
public static class LeNet {

    /// <summary>
    /// Supported LeNet versions
    /// </summary>
    public enum Version {
        V1 = 1,
        V5 = 5,
        /// <summary>
        /// Latest supported version
        /// </summary>
        Latest = 5
    }

    /// <summary>
    /// Typical width (in pixels) for an image processed by LeNet
    /// </summary>
    const int IMG_WIDTH = 28;
    /// <summary>
    /// Typical height (in pixels) for an image processed by LeNet
    /// </summary>
    const int IMG_HEIGHT = 28;
    /// <summary>
    /// Typical number of channels for an image processed by LeNet 
    /// </summary>
    const int IMG_CHANNELS = 1;


    /// <summary>
    /// Construct a LeNet network
    /// </summary>
    /// <param name="version">network architecture version</param>
    /// <param name="output_classes">number of output classifications</param>
    /// <param name="activation">activation function</param>
    /// <returns>network</returns>
    /// <exception cref="ArgumentException">thrown when an unsupported version is supplied</exception>
    public static ConvolutionalFeedforwardNetwork Make(Version version, int output_classes, int img_width = IMG_WIDTH, int img_height = IMG_HEIGHT, ActivationFunction? activation = null) {
        return version switch {
            Version.V1 => MakeV1(output_classes, img_width, img_height, activation),
            Version.V5 => MakeV5(output_classes, img_width, img_height, activation),
            _ => throw new ArgumentException(nameof(version))
        };
    }

    private static ConvolutionalFeedforwardNetwork MakeV1(int output_classes, int img_width, int img_height, ActivationFunction? activation) {
        activation = activation ?? ReLU.Instance;

        return new ConvolutionalFeedforwardNetwork(
            new ConvolutionLayer(
                new Shape3D(IMG_CHANNELS, img_height, img_width),
                padding: Padding.Valid, 
                stride: 1, 
                filters: ConvolutionFilter.Make(12, 1, 5)
            )
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => 
                new LocalAvgPoolingLayer(
                    ishape,
                    size: 2,
                    stride: 2
                )
            )
            .Then(ishape => 
                new ConvolutionLayer(
                    ishape,
                    padding: Padding.Valid,
                    stride: 1,
                    filters: ConvolutionFilter.Make(8, 1, 5)
                )
            )
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => 
                new LocalAvgPoolingLayer(
                    ishape,
                    size: 2,
                    stride: 2
                )
            )
            .Then(ishape => 
                new FullyConnectedLayer(
                    ishape.Count,
                    output_classes
                )
            )
            .Then(ishape => new SoftmaxLayer(output_classes))
        );
    }

    private static ConvolutionalFeedforwardNetwork MakeV5(int output_classes, int img_width, int img_height, ActivationFunction? activation) {
        activation = activation ?? ReLU.Instance;

        return new ConvolutionalFeedforwardNetwork(
            new ConvolutionLayer(
                new Shape3D(IMG_CHANNELS, img_height, img_width),
                padding: Padding.Valid, 
                stride: 1, 
                filters: ConvolutionFilter.Make(6, 1, 5)
            )
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => 
                new LocalAvgPoolingLayer (
                    ishape,
                    size: 2, 
                    stride: 2
                )
            ).Then(ishape => 
                new ConvolutionLayer(
                    ishape,
                    padding: Padding.Valid, 
                    stride: 1, 
                    filters: ConvolutionFilter.Make(16, 6, 5)
                )
            )
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => 
                new LocalAvgPoolingLayer(
                    ishape,
                    size: 2, 
                    stride: 2
                )
            ).Then(ishape =>
                new FullyConnectedLayer(
                    input_size: ishape.Count, 
                    neurons: 120
                )
            )
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => 
                new FullyConnectedLayer(
                    input_size: ishape.Count, 
                    neurons: 84
                )
            )
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => 
                new FullyConnectedLayer(
                    input_size: ishape.Count, 
                    neurons: output_classes
                )
            )
            .Then(ishape => new SoftmaxLayer(output_classes))
        );
    }

}