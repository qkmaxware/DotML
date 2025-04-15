namespace DotML.Network;

/// <summary>
/// Utility methods related to the ESPCN (Efficient Sub-pixel Convolutional Network) architecture
/// </summary>
public static class ESPCN {
    /// <summary>
    /// Supported ESPCN versions
    /// </summary>
    public enum Version {
        V1 = 1,
        /// <summary>
        /// Latest supported version
        /// </summary>
        Latest = 1
    }

    /// <summary>
    /// Typical number of channels per image processed by ESPCN (typically three, RGB)
    /// </summary>
    public const int IMG_CHANNELS = 3;
    /// <summary>
    /// Typical width (in pixels) for an image processed by ESPCN (typically 32 pixels)
    /// </summary>
    public const int IMG_WIDTH = 32;
    /// <summary>
    /// Typical height (in pixels) for an image processed by ESPCN (typically 32 pixels)
    /// </summary>
    public const int IMG_HEIGHT = 32;
    /// <summary>
    /// Typical upscaling factor for an image processed by ESPCN (typically 3x)
    /// </summary>
    public const int UPSCALING_FACTOR = 3;
    
    public static FeedforwardNetwork Make(Version version, int img_channels = IMG_CHANNELS, int img_width = IMG_WIDTH, int img_height = IMG_HEIGHT, int scaling = UPSCALING_FACTOR, ActivationFunction? activation = null) {
        scaling = Math.Max(1, scaling);

        var net = version switch {
            Version.V1 => MakeV1(img_channels, img_width, img_height, scaling, activation),
            _ => throw new ArgumentException(nameof(version))
        };
        net.Name = "ESPCN-v" + ((int)version);

        return net;
    }

    private static FeedforwardNetwork MakeV1(int img_channels, int img_width, int img_height, int scaling, ActivationFunction? activation) {
        activation = activation ?? ReLU.Instance;

        return new FeedforwardNetwork(
            new ConvolutionLayer(
                input_size: new Shape3D(img_channels, img_height, img_width), 
                padding: Padding.Same,
                filters: ConvolutionFilter.Make(filters: 64, kernels_per_filter: img_channels, 5)
            )
            .WithActivation(activation)
            .Then((ishape) => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                filters: ConvolutionFilter.Make(filters: 32, kernels_per_filter: ishape.Channels, 3)
            ))
            .WithActivation(activation)
            .Then((ishape) => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                filters: ConvolutionFilter.Make(filters: img_channels * scaling * scaling, kernels_per_filter: ishape.Channels, 3)
            ))
            .Then((ishape) => new PixelShuffle(
                input_size: ishape,
                upscale_factor: scaling
            ))
        );
    }
}