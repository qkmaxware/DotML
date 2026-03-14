namespace DotML.Network.Templates;

/// <summary>
/// Factory for creating the ESPCN (Efficient Sub-pixel Convolutional Network) architecture
/// </summary>
public class ESPCNFactory
: INetworkModuleFactory<ESPCNFactory.BuildSettings>
{
    /// <summary>
    /// Typical number of channels per image processed by ESPCN (typically 1, Y/Luminance)
    /// </summary>
    public const int IMG_CHANNELS = 1;
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

    public class BuildSettings
    {
        public int ImgChannels { get; set; } = IMG_CHANNELS;
        public int ImgWidth { get; set; } = IMG_WIDTH;
        public int ImgHeight { get; set; } = IMG_HEIGHT;
        public int UpscalingFactor { get; set; } = UPSCALING_FACTOR;
        public ActivationFunction? Activation { get; set; } = null;

        public BuildSettings() { }
        public BuildSettings(int imgChannels = IMG_CHANNELS, int imgWidth = IMG_WIDTH, int imgHeight = IMG_HEIGHT, int upscalingFactor = UPSCALING_FACTOR, ActivationFunction? activation = null)
        {
            ImgChannels = imgChannels;
            ImgWidth = imgWidth;
            ImgHeight = imgHeight;
            UpscalingFactor = upscalingFactor;
            Activation = activation;
        }
    }

    public INetworkModule Make(BuildSettings settings)
    {
        var scaling = Math.Max(1, settings.UpscalingFactor);
        var activation = settings.Activation ?? HyperbolicTangent.Instance;

        var ishape = new Shape(settings.ImgChannels, settings.ImgHeight, settings.ImgWidth);

        return new ArchitectureBlock(
            name: "ESPCN",
            inputShape: ishape,
            rootModule: SequentialBlock
            .Begin(ishape)
            .Then((ishape) => new Conv2D(
                outChannels: 64,
                inChannelsPerGroup: ishape.Length(^3), // [C, H, W]
                groups: 1,
                kernel: (5, 5),
                stride: (1, 1),
                dilation: (1, 1),
                padding: Padding.Same.ToTuple(kernel: (5, 5), stride: (1, 1), dilation: (1, 1)) // Same padding
            ))
            .WithActivation(activation)
            .Then((ishape) => new Conv2D(
                outChannels: 32,
                inChannelsPerGroup: ishape.Length(^3), // [C, H, W]
                groups: 1,
                kernel: (3, 3),
                stride: (1, 1),
                dilation: (1, 1),
                padding: Padding.Same.ToTuple(kernel: (3, 3), stride: (1, 1), dilation: (1, 1)) // Same padding
            ))
            .WithActivation(activation)
            .Then((ishape) => new Conv2D(
                outChannels: settings.ImgChannels * (int)Math.Pow(scaling, 2),
                inChannelsPerGroup: ishape.Length(^3), // [C, H, W]
                groups: 1,
                kernel: (3, 3),
                stride: (1, 1),
                dilation: (1, 1),
                padding: Padding.Same.ToTuple(kernel: (3, 3), stride: (1, 1), dilation: (1, 1)) // Same padding
            ))
            .Then((ishape) => new PixelShuffler(
                upscale: scaling
            ))
            .Finalize()
        );
    }

}