namespace DotML.Network;

/// <summary>
/// Utility methods related to the FSRCNN (Fast Super-Resolution Convolutional Neural Network) architecture
/// </summary>
public static class FSRCNN {

    /// <summary>
    /// Supported FSRCNN versions
    /// </summary>
    public enum Version {
        V1 = 1,
        /// <summary>
        /// Latest supported version
        /// </summary>
        Latest = 1
    }

    /// <summary>
    /// Typical number of channels per image processed by FSRCNN (typically three, RGB)
    /// </summary>
    public const int IMG_CHANNELS = 3;
    /// <summary>
    /// Typical width (in pixels) for an image processed by FSRCNN (typically 32 pixels)
    /// </summary>
    public const int IMG_WIDTH = 32;
    /// <summary>
    /// Typical height (in pixels) for an image processed by FSRCNN (typically 32 pixels)
    /// </summary>
    public const int IMG_HEIGHT = 32;
    /// <summary>
    /// Typical upscaling factor for an image processed by FSRCNN (typically 2x)
    /// </summary>
    public const int UPSCALING_FACTOR = 2;

    public static FeedforwardNetwork Make(Version version, int img_channels = IMG_CHANNELS, int img_width = IMG_WIDTH, int img_height = IMG_HEIGHT, int scaling = UPSCALING_FACTOR, ActivationFunction? activation = null) {
        scaling = Math.Max(1, scaling);

        var net = version switch {
            Version.V1 => MakeV1(img_channels, img_width, img_height, scaling, activation),
            _ => throw new ArgumentException(nameof(version))
        };
        net.Name = "FSRCNN-v" + ((int)version);

        return net;
    }

    private static FeedforwardNetwork MakeV1(int img_channels, int img_width, int img_height, int scaling, ActivationFunction? activation) {
        activation = activation ?? ReLU.Instance;
        var features_to_recognize = 56;

        var kernel_height = (scaling * img_height) - (img_height - 1) * scaling;
        var kernel_width = (scaling * img_width) - (img_width - 1) * scaling;

        return new FeedforwardNetwork(
            // Initial feature extraction layer
            new ConvolutionLayer(
                input_size: new Shape3D(img_channels, img_height, img_width),
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(filters: features_to_recognize, kernels_per_filter: img_channels, kernel_size: 5)
            )
            .WithActivation(activation)
            // Feature learning layers
            .Then((ishape) => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(filters: features_to_recognize, kernels_per_filter: ishape.Channels, kernel_size: 3)
            ))
            .WithActivation(activation)
            .Then((ishape) => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(filters: features_to_recognize, kernels_per_filter: ishape.Channels, kernel_size: 3)
            ))
            .WithActivation(activation)
            .Then((ishape) => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(filters: features_to_recognize, kernels_per_filter: ishape.Channels, kernel_size: 3)
            ))
            .WithActivation(activation)
            .Then((ishape) => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(filters: features_to_recognize, kernels_per_filter: ishape.Channels, kernel_size: 3)
            ))
            .WithActivation(activation)
            // Upscaling (de-convolution) layer
            .Then((ishape) => new TransposeConvolutionLayer(
                input_size: ishape,
                inputPaddingX: 0, inputPaddingY: 0,
                outputPaddingX: 0, outputPaddingY: 0,
                strideX: scaling, strideY: scaling,
                filters: ConvolutionFilter.Make(filters: features_to_recognize, kernels_per_filter: ishape.Channels, kernel_rows: kernel_height, kernel_columns: kernel_width)
            ))
            // Final output layer
            .Then((ishape) => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(filters: img_channels, kernels_per_filter: ishape.Channels, kernel_size: 3)
            ))
        );
    }

}