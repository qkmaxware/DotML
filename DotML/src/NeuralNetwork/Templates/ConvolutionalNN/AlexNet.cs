namespace DotML.Network.Templates;

/// <summary>
/// Factory for creating the AlexNet Convolutional Neural Network architecture
/// </summary>
public class AlexNetFactory
: INetworkModuleFactory<AlexNetFactory.BuildSettings>
{
    /// <summary>
    /// Typical number of output classes used by AlexNet (typically 1000)
    /// </summary>
    public const int OUT_CLASSES = 100;
    /// <summary>
    /// Typical number of channels per image processed by AlexNet (typically three, RGB)
    /// </summary>
    public const int IMG_CHANNELS = 3;
    /// <summary>
    /// Typical width (in pixels) for an image processed by AlexNet (typically 227 pixels)
    /// </summary>
    public const int IMG_WIDTH = 227;
    /// <summary>
    /// Typical height (in pixels) for an image processed by AlexNet (typically 227 pixels)
    /// </summary>
    public const int IMG_HEIGHT = 227;

    /// <summary>
    /// Settings for an AlexNet v1 network
    /// </summary>
    public class BuildSettings
    {
        public int OutputClasses { get; set; } = OUT_CLASSES;
        public int ImgChannels { get; set; } = IMG_CHANNELS;
        public int ImgWidth { get; set; } = IMG_WIDTH;
        public int ImgHeight { get; set; } = IMG_HEIGHT;
        public ActivationFunction? Activation { get; set; } = null;
        public bool NormalizeLayers { get; set; } = false;

        public BuildSettings() { }
        public BuildSettings(int outputClasses, int imgChannels = IMG_CHANNELS, int imgWidth = IMG_WIDTH, int imgHeight = IMG_HEIGHT, ActivationFunction? activation = null, bool normalizeLayers = false)
        {
            OutputClasses = outputClasses;
            ImgChannels = imgChannels;
            ImgWidth = imgWidth;
            ImgHeight = imgHeight;
            Activation = activation;
            NormalizeLayers = normalizeLayers;
        }
    }

    public INetworkModule MakeDefault() => Make(new BuildSettings());

    public INetworkModule Make(BuildSettings settings)
    {
        var activation = settings.Activation ?? ReLU.Instance;
        double scalingFactor = Math.Max(1, (settings.ImgWidth * settings.ImgHeight) / (double)(IMG_WIDTH * IMG_HEIGHT));
        var neurons = Math.Max(4096, (int)(4096 * scalingFactor));

        var ishape = new TensorShape(settings.ImgChannels, settings.ImgHeight, settings.ImgWidth);

        return new ArchitectureBlock(
            name: "AlexNet",
            inputShape: ishape,
            rootModule: SequentialBlock
            .Begin(ishape)
            // Convo 1
            .Then(ishape => new Conv2D(
                outChannels: 96,
                inChannelsPerGroup: ishape.Length(^3),
                groups: 1,
                kernel: (11, 11),
                stride: (4, 4),
                dilation: (1, 1),
                padding: (0, 0, 0, 0)
            ))
            .Then(ishape => new Activation(activation))
            .ThenIf(settings.NormalizeLayers, (ishape) => new LayerNorm2(ishape.Length(^3), ishape.Length(^2), ishape.Length(^1)))
            .Then(ishape => new MaxPool2D(
                size: 3,
                stride: 2,
                padding: 0
            ))
            // Convo 2
            .Then(ishape => new Conv2D(
                outChannels: 256,
                inChannelsPerGroup: ishape.Length(^3),
                groups: 1,
                kernel: (5, 5),
                stride: (1, 1),
                dilation: (1, 1),
                padding: (2, 2, 2, 2) // 'Same' padding for 5x5
            ))
            .Then(ishape => new Activation(activation))
            .ThenIf(settings.NormalizeLayers, ishape => new LayerNorm2(ishape.Length(^3), ishape.Length(^2), ishape.Length(^1)))
            .Then(ishape => new MaxPool2D(
                size: 3,
                stride: 2,
                padding: 0
            ))
            // Set of 3 Convo
            .Then(ishape => new Conv2D(
                outChannels: 384,
                inChannelsPerGroup: ishape.Length(^3),
                groups: 1,
                kernel: (3, 3),
                stride: (1, 1),
                dilation: (1, 1),
                padding: (1, 1, 1, 1) // 'Same' padding for 3x3
            ))
            .Then(ishape => new Activation(activation))
            .Then(ishape => new Conv2D(
                outChannels: 384,
                inChannelsPerGroup: ishape.Length(^3),
                groups: 1,
                kernel: (3, 3),
                stride: (1, 1),
                dilation: (1, 1),
                padding: (1, 1, 1, 1)
            ))
            .Then(ishape => new Activation(activation))
            .Then(ishape => new Conv2D(
                outChannels: 256,
                inChannelsPerGroup: ishape.Length(^3),
                groups: 1,
                kernel: (3, 3),
                stride: (1, 1),
                dilation: (1, 1),
                padding: (1, 1, 1, 1)
            ))
            .Then(ishape => new Activation(activation))
            .ThenIf(settings.NormalizeLayers, ishape => new LayerNorm2(ishape.Length(^3), ishape.Length(^2), ishape.Length(^1)))
            .Then(ishape => new MaxPool2D(
                size: 3,
                stride: 2,
                padding: 0
            ))
            // Flattening and Dense layers
            .Then(ishape => new DenseLinear(
                ishape.LogicalElementCount(),
                neurons
            ))
            .ThenIf(settings.NormalizeLayers, ishape => new LayerNorm2(ishape.Length(^3), ishape.Length(^2), ishape.Length(^1)))
            .Then(ishape => new Activation(activation))
            .Then(ishape => new DenseLinear(
                ishape.LogicalElementCount(),
                neurons
            ))
            .ThenIf(settings.NormalizeLayers, ishape => new LayerNorm2(ishape.Length(^3), ishape.Length(^2), ishape.Length(^1)))
            .Then(ishape => new Activation(activation))
            .Then(ishape => new DenseLinear(
                ishape.LogicalElementCount(),
                settings.OutputClasses
            ))
            .Then(ishape => new SoftmaxOutput())
            .Finalize()
        );
    }
}