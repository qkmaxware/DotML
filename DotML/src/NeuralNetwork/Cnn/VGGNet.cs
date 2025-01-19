namespace DotML.Network;

/// <summary>
/// Utility methods related to the VGG-Net Convolutional Neural Network architecture
/// </summary>
public static class VGGNet {

    /// <summary>
    /// Supported VGGNet versions
    /// </summary>
    public enum Version {
        VGG16 = 16,
        /// <summary>
        /// Latest supported version
        /// </summary>
        Latest = 16
    }

    /// <summary>
    /// Typical number of channels per image processed by VGGNet (typically three, RGB)
    /// </summary>
    const int IMG_CHANNELS = 3;
    /// <summary>
    /// Typical width (in pixels) for an image processed by VGGNet (typically 224 pixels)
    /// </summary>
    const int IMG_WIDTH = 224;
    /// <summary>
    /// Typical height (in pixels) for an image processed by VGGNet (typically 224 pixels)
    /// </summary>
    const int IMG_HEIGHT = 224;

    /// <summary>
    /// Construct an VGGNet network
    /// </summary>
    /// <param name="version">network architecture version</param>
    /// <param name="output_classes">number of output classifications</param>
    /// <param name="activation">activation function</param>
    /// <returns>network</returns>
    /// <exception cref="ArgumentException">thrown when an unsupported version is supplied</exception>
    public static FeedforwardNetwork Make(Version version, int output_classes, int img_channels = IMG_CHANNELS, int img_width = IMG_WIDTH, int img_height = IMG_HEIGHT, ActivationFunction? activation = null) {
        var net = version switch {
            Version.VGG16 => MakeVGG_16(output_classes, img_channels, img_width, img_height, activation),
            _ => throw new ArgumentException(nameof(version))
        };
        net.Name = "VGG-" + ((int)version);
        return net;
    }

    private static FeedforwardNetwork MakeVGG_16(int output_classes, int img_channels, int img_width, int img_height, ActivationFunction? activation) {
        
        activation = activation ?? ReLU.Instance;
        Shape3D input = new Shape3D(img_channels, img_height, img_width);

        double scalingFactor = Math.Max(1, (img_width * img_height) / (double)(IMG_WIDTH * IMG_HEIGHT));
        var neurons = Math.Max(4096, (int)(4096 * scalingFactor));

        return new FeedforwardNetwork(
            // First block
            new ConvolutionLayer(
                input_size: input,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(64, input.Channels, 3)
            )
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(64, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new LocalMaxPoolingLayer(ishape, size: 2, stride: 2))
            // Second block
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(128, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(128, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new LocalMaxPoolingLayer(ishape, size: 2, stride: 2))
            // Third block
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(256, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(256, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(256, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new LocalMaxPoolingLayer(ishape, size: 2, stride: 2))
            // Fourth block
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(512, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(512, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(512, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new LocalMaxPoolingLayer(ishape, size: 2, stride: 2))
            // Fifth block
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(512, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(512, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(512, ishape.Channels, 3)
            ))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new LocalMaxPoolingLayer(ishape, size: 2, stride: 2))
            // FCs
            .Then(ishape => new FullyConnectedLayer(ishape.Count, neurons))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new DropoutLayer(ishape))
            .Then(ishape => new FullyConnectedLayer(ishape.Count, neurons))
            .Then(ishape => new ActivationLayer(ishape, activation))
            .Then(ishape => new DropoutLayer(ishape))
            // Output layer
            .Then(ishape => new FullyConnectedLayer(ishape.Count, output_classes))
            .Then(ishape => new SoftmaxLayer(ishape.Count))
        );
    }

}