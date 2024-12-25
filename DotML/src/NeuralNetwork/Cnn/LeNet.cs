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
        /// <summary>
        /// Latest supported version
        /// </summary>
        Latest = 1
    }

    /// <summary>
    /// Construct a LeNet network
    /// </summary>
    /// <param name="version">network architecture version</param>
    /// <param name="output_classes">number of output classifications</param>
    /// <param name="activation">activation function</param>
    /// <returns>network</returns>
    /// <exception cref="ArgumentException">thrown when an unsupported version is supplied</exception>
    public static ConvolutionalFeedforwardNetwork Make(Version version, int output_classes, ActivationFunction? activation = null) {
        return version switch {
            Version.V1 => MakeV1(output_classes, activation),
            _ => throw new ArgumentException(nameof(version))
        };
    }

    /// <summary>
    /// Construct a network using the LeNet architecture
    /// </summary>
    /// <param name="output_classes">number of output classifications</param>
    /// <returns>network</returns>
    private static ConvolutionalFeedforwardNetwork MakeV1(int output_classes, ActivationFunction? activation = null) {
        const int IMG_WIDTH = 32;
        const int IMG_HEIGHT = 32;
        const int IMG_CHANNELS = 1;

        activation = activation ?? ReLU.Instance;

        return new ConvolutionalFeedforwardNetwork(
            new ConvolutionLayer(
                new Shape3D(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH),
                padding: Padding.Valid, 
                stride: 1, 
                filters: ConvolutionFilter.Make(6, 1, 5)
            ),
            new ActivationLayer(new Shape3D(6, 28, 28), activation),
            new LocalAvgPoolingLayer (
                new Shape3D(6, 28, 28),
                size: 2, 
                stride: 2
            ),
            new ConvolutionLayer(
                new Shape3D(6, 14, 14),
                padding: Padding.Valid, 
                stride: 1, 
                filters: ConvolutionFilter.Make(16, 6, 5)
            ),
            new ActivationLayer(new Shape3D(16, 10, 10), activation),
            new LocalAvgPoolingLayer(
                new Shape3D(16, 10, 10),
                size: 2, 
                stride: 2
            ),
            new FullyConnectedLayer(
                input_size: 400, 
                neurons: 120
            ),
            new ActivationLayer(new Shape3D(1, 120, 1), activation),
            new FullyConnectedLayer(
                input_size: 120, 
                neurons: 84
            ),
            new ActivationLayer(new Shape3D(1, 84, 1), activation),
            new FullyConnectedLayer(
                input_size: 84, 
                neurons: output_classes
            ),
            new SoftmaxLayer(output_classes)
        );
    }

}