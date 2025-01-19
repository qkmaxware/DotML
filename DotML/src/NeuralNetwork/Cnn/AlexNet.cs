namespace DotML.Network;

/// <summary>
/// Utility methods related to the AlexNet Convolutional Neural Network architecture
/// </summary>
public static class AlexNet {

    /// <summary>
    /// Supported AlexNet versions
    /// </summary>
    public enum Version {
        V1 = 1,
        /// <summary>
        /// Latest supported version
        /// </summary>
        Latest = 1
    }

    /// <summary>
    /// Typical number of channels per image processed by AlexNet (typically three, RGB)
    /// </summary>
    const int IMG_CHANNELS = 3;
    /// <summary>
    /// Typical width (in pixels) for an image processed by AlexNet (typically 227 pixels)
    /// </summary>
    const int IMG_WIDTH = 227;
    /// <summary>
    /// Typical height (in pixels) for an image processed by AlexNet (typically 227 pixels)
    /// </summary>
    const int IMG_HEIGHT = 227;

    /// <summary>
    /// Construct an AlexNet network
    /// </summary>
    /// <param name="version">network architecture version</param>
    /// <param name="output_classes">number of output classifications</param>
    /// <param name="activation">activation function</param>
    /// <returns>network</returns>
    /// <exception cref="ArgumentException">thrown when an unsupported version is supplied</exception>
    public static FeedforwardNetwork Make(Version version, int output_classes, int img_channels = IMG_CHANNELS, int img_width = IMG_WIDTH, int img_height = IMG_HEIGHT, ActivationFunction? activation = null) {
        var net = version switch {
            Version.V1 => MakeV1(output_classes, img_channels, img_width, img_height, activation),
            _ => throw new ArgumentException(nameof(version))
        };
        net.Name = "AlexNet-v" + ((int)version);
        return net;
    }

    private static FeedforwardNetwork MakeV1(int output_classes, int img_channels, int img_width, int img_height, ActivationFunction? activation) {
        activation = activation ?? ReLU.Instance;

        double scalingFactor = Math.Max(1, (img_width * img_height) / (double)(IMG_WIDTH * IMG_HEIGHT));
        var neurons = Math.Max(4096, (int)(4096 * scalingFactor));

        return new FeedforwardNetwork(
            // Convo 1
            new ConvolutionLayer(
                new Shape3D(img_channels, img_height, img_width),
                padding: Padding.Valid, 
                stride: 4, 
                filters: ConvolutionFilter.Make(96, img_channels, 11)
            )
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new LocalMaxPoolingLayer(ishape, stride: 2, size: 3))
            // Convo 2
            .Then(ishape => new ConvolutionLayer(
                ishape,
                padding: Padding.Same, 
                stride: 1, 
                filters: ConvolutionFilter.Make(256, ishape.Channels, 5)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new LocalMaxPoolingLayer(ishape, stride: 2, size: 3))
            // Set of 3 Convo
            .Then(ishape => new ConvolutionLayer(
                ishape,
                padding: Padding.Same, 
                stride: 1, 
                filters: ConvolutionFilter.Make(384, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new ConvolutionLayer(
                ishape,
                padding: Padding.Same, 
                stride: 1, 
                filters: ConvolutionFilter.Make(384, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new ConvolutionLayer(
                ishape,
                padding: Padding.Same, 
                stride: 1, 
                filters: ConvolutionFilter.Make(256, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new LocalMaxPoolingLayer(ishape, stride: 2, size: 3))
            // Flattening
            .Then(ishape => new FullyConnectedLayer(ishape.Count, neurons))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new FullyConnectedLayer(ishape.Count, neurons))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new FullyConnectedLayer(ishape.Count, output_classes))
            .Then(ishape => new SoftmaxLayer(ishape.Count))
        );
    }
}