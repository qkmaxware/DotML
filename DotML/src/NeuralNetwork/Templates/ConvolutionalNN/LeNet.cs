namespace DotML.Network.Templates;

/// <summary>
/// Factory for creating the LeNet Convolutional Neural Network architecture
/// </summary>
public class LeNetFactory
: INetworkModuleFactory<LeNetFactory.BuildSettingsV1>
, INetworkModuleFactory<LeNetFactory.BuildSettingsV5>
, INetworkModuleFactory<LeNetFactory.BuildSettingsModern>
{

    /// <summary>
    /// Typical number of output classes used by LeNet (typically 10)
    /// </summary>
    public const int OUT_CLASSES = 10;
    /// <summary>
    /// Typical number of channels per image processed by LeNet (typically one, grayscale)
    /// </summary>
    public const int IMG_CHANNELS = 1;
    /// <summary>
    /// Typical width (in pixels) for an image processed by LeNet (typically 28 pixels)
    /// </summary>
    public const int IMG_WIDTH = 28;
    /// <summary>
    /// Typical height (in pixels) for an image processed by LeNet (typically 28 pixels)
    /// </summary>
    public const int IMG_HEIGHT = 28;

    /// <summary>
    /// Settings for a LeNet v1 network
    /// </summary>
    public class BuildSettingsV1
    {
        public int OutputClasses { get; set; } = OUT_CLASSES;
        public int ImgChannels { get; set; } = IMG_CHANNELS;
        public int ImgWidth { get; set; } = IMG_WIDTH;
        public int ImgHeight { get; set; } = IMG_HEIGHT;
        public ActivationFunction? Activation { get; set; } = null;

        public BuildSettingsV1() { }
        public BuildSettingsV1(int outputClasses, int imgChannels = IMG_CHANNELS, int imgWidth = IMG_WIDTH, int imgHeight = IMG_HEIGHT, ActivationFunction? activation = null) {
            OutputClasses = outputClasses;
            ImgChannels = imgChannels;
            ImgWidth = imgWidth;
            ImgHeight = imgHeight;
            Activation = activation;
        }
    }

    /// <summary>
    /// Settings for a LeNet v5 network
    /// </summary>
    public class BuildSettingsV5
    {
        public int OutputClasses { get; set; } = OUT_CLASSES;
        public int ImgChannels { get; set; } = IMG_CHANNELS;
        public int ImgWidth { get; set; } = IMG_WIDTH;
        public int ImgHeight { get; set; } = IMG_HEIGHT;
        public ActivationFunction? Activation { get; set; } = null;

        public BuildSettingsV5() { }
        public BuildSettingsV5(int outputClasses, int imgChannels = IMG_CHANNELS, int imgWidth = IMG_WIDTH, int imgHeight = IMG_HEIGHT, ActivationFunction? activation = null) {
            OutputClasses = outputClasses;
            ImgChannels = imgChannels;
            ImgWidth = imgWidth;
            ImgHeight = imgHeight;
            Activation = activation;
        }
    }

    /// <summary>
    /// Settings for a modern update to LeNet-v5
    /// </summary>
    public class BuildSettingsModern
    {
        public int ImgWidth { get; set; } = IMG_WIDTH;
        public int ImgHeight { get; set; } = IMG_HEIGHT;
        public int ImgChannels { get; set; } = 3;
        public int OutputClasses { get; set; } = OUT_CLASSES;

        public ActivationFunction Activation { get; set; } = ReLU.Instance;

        public int Conv1Channels { get; set; } = 32;
        public int Conv2Channels { get; set; } = 64;
        public Padding Padding {get; set;} = Padding.Same;

        public int FullyConnectedUnits { get; set; } = 256;
        public float Dropout { get; set; } = 0.25f;

        public bool UseBatchNorm { get; set; } = true;
        public bool UseMaxPool { get; set; } = true;
    }

    public INetworkModule MakeDefault() => Make(new BuildSettingsV5());

    public INetworkModule Make(BuildSettingsV1 settings)
    {
        var activation = settings.Activation ?? ReLU.Instance;

        var ishape = new TensorShape(settings.ImgChannels, settings.ImgHeight, settings.ImgWidth);

        return new ArchitectureBlock(
            name: "LeNetV1",
            inputShape: ishape,
            SequentialBlock
            .Begin(ishape)
            .Then((ishape) => new Conv2D(
                outChannels: 12,
                inChannelsPerGroup: ishape.Length(^3), // [C, H, W]
                groups: 1,
                kernel: (5, 5),
                stride: (1, 1),
                dilation: (1, 1),
                padding: (0, 0, 0, 0) // Valid padding
            ))
            .Then((ishape) => new Activation(activation))
            .Then((ishape) => new AvgPool2D(
                size: 2,
                stride: 2,
                padding: 0
            ))
            .Then((ishape) => new Conv2D(
                outChannels: 8,
                inChannelsPerGroup: ishape.Length(^3), // [C, H, W]
                groups: 1,
                kernel: (5, 5),
                stride: (1, 1),
                dilation: (1, 1),
                padding: (0, 0, 0, 0) // Valid padding
            ))
            .Then((ishape) => new Activation(activation))
            .Then(ishape => new AvgPool2D(
                    size: 2,
                    stride: 2,
                    padding: 0
            ))
            .Then((ishape) => new DenseLinear(
                ishape.LogicalElementCount(),
                settings.OutputClasses
            ))
            //.Then(ishape => new SoftmaxOutput())
            .Finalize()
        );
    }

    public INetworkModule Make(BuildSettingsV5 settings)
    {
        var img_width               = settings.ImgWidth;
        var img_height              = settings.ImgHeight;
        var activation              = settings.Activation ?? ReLU.Instance;

        double scalingFactor        = Math.Max(1, (img_width * img_height) / (double)(IMG_WIDTH * IMG_HEIGHT));

        int fullyConnectedNeurons1  = Math.Max(120, (int)(120 * scalingFactor));
        int fullyConnectedNeurons2  = Math.Max(84, (int)(84 * scalingFactor));

        var ishape = new TensorShape(settings.ImgChannels, settings.ImgHeight, settings.ImgWidth);

        return new ArchitectureBlock(
            name: "LeNetV5",
            inputShape: ishape,
            rootModule: SequentialBlock
            .Begin(ishape)
            .Then((ishape) => new Conv2D(
                outChannels: 6,
                inChannelsPerGroup: ishape.Length(^3), // [C, H, W]
                groups: 1,
                kernel: (5, 5),
                stride: (1, 1),
                dilation: (1, 1),
                padding: (0, 0, 0, 0) // Valid padding
            ))
            .Then((ishape) => new Activation(activation))
            .Then((ishape) => new AvgPool2D(
                size: 2,
                stride: 2,
                padding: 0
            ))
            .Then((ishape) => new Conv2D(
                outChannels: 16,
                inChannelsPerGroup: ishape.Length(^3), // [C, H, W]
                groups: 1,
                kernel: (5, 5),
                stride: (1, 1),
                dilation: (1, 1),
                padding: (0, 0, 0, 0) // Valid padding
            ))
            .Then((ishape) => new Activation(activation))
            .Then((ishape) => new AvgPool2D(
                size: 2,
                stride: 2,
                padding: 0
            ))
            .Then((ishape) => new DenseLinear(
                ishape.LogicalElementCount(),
                fullyConnectedNeurons1
            ))
            .Then(ishape => new Activation(activation))
            .Then((ishape) => new DenseLinear(
                ishape.LogicalElementCount(),
                fullyConnectedNeurons2
            ))
            .Then(ishape => new Activation(activation))
            .Then((ishape) => new DenseLinear(
                ishape.LogicalElementCount(),
                settings.OutputClasses
            ))
            //.Then(ishape => new SoftmaxOutput()) // No longer required, just use the correct loss function to train logits
            .Finalize()
        );
    }

    public INetworkModule Make(BuildSettingsModern settings)
    {
        var activation = settings.Activation ?? ReLU.Instance;

        var ishape = new TensorShape(
            settings.ImgChannels,
            settings.ImgHeight,
            settings.ImgWidth
        );

        return new ArchitectureBlock(
            name: "LeNet++",
            inputShape: ishape,
            rootModule: SequentialBlock
            .Begin(ishape)

            // --------------------
            // Conv Block 1
            // --------------------
            .Then(s => new Conv2D(
                outChannels: settings.Conv1Channels,
                inChannelsPerGroup: s.Length(^3),
                groups: 1,
                kernel: (5, 5),
                stride: (1, 1),
                dilation: (1, 1),
                padding: settings.Padding.ToPadding((5, 5), (1, 1), (1, 1))   // SAME padding for CIFAR-10
            ))
            .ThenIf(settings.UseBatchNorm, (s) => new BatchNorm2D(s.Length(^3)))
            .Then(s => new Activation(activation))
            .Then(s => settings.UseMaxPool
                ? new MaxPool2D(size: 2, stride: 2, padding: 0)
                : new AvgPool2D(size: 2, stride: 2, padding: 0))

            // --------------------
            // Conv Block 2
            // --------------------
            .Then(s => new Conv2D(
                outChannels: settings.Conv2Channels,
                inChannelsPerGroup: s.Length(^3),
                groups: 1,
                kernel: (5, 5),
                stride: (1, 1),
                dilation: (1, 1),
                padding: settings.Padding.ToPadding((5, 5), (1, 1), (1, 1))
            ))
            .ThenIf(settings.UseBatchNorm, (s) => new BatchNorm2D(s.Length(^3)))
            .Then(s => new Activation(activation))
            .Then(s => settings.UseMaxPool
                ? new MaxPool2D(size: 2, stride: 2, padding: 0)
                : new AvgPool2D(size: 2, stride: 2, padding: 0))

            // --------------------
            // Dense Blocks
            // --------------------
            .Then(s => new Dropout(settings.Dropout))

            .Then(s => new DenseLinear(
                s.LogicalElementCount(),     // flatten
                settings.FullyConnectedUnits
            ))
            .Then(s => new Activation(activation))

            .Then(s => new DenseLinear(
                s.LogicalElementCount(),
                settings.OutputClasses
            ))

            .Finalize()
        );
    }
}