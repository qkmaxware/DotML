namespace DotML.Network.Templates;

/// <summary>
/// Factory for creating the LeNet Convolutional Neural Network architecture
/// </summary>
public class UNetFactory : INetworkModuleFactory<UNetFactory.BuildSettings>
{
    public class BuildSettings
    {
        public int InputChannels { get; set; } = 3;      // e.g., RGB or RGB + timestep
        public int OutputChannels { get; set; } = 3;     // RGB
        public int BaseFeatureCount { get; set; } = 64;  // Number of channels in first layer
        public int Depth { get; set; } = 3;              // Number of encoder/decoder stages
        public int ImageWidth { get; set; } = 32;             // Optional, used for padding calc if needed must be power of 2 and larger than 2^depth for equal input and output image sizes
        public int ImageHeight { get; set; } = 32;
        public bool UseNormalization { get; set; } = true;
        public ActivationFunction Activation { get; set; } = ActivationFunctions.ReLU;
    }

    private static SequentialBlock ConvBlock(int inChannels, int outChannels, ActivationFunction fn, bool useNormalization)
    {
        if (useNormalization)
        {
            return new SequentialBlock([
                new Conv2D(outChannels, inChannels, 1, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1)),
                new BatchNorm2D(outChannels),
                new Activation(fn),
                new Conv2D(outChannels, outChannels, 1, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1)),
                new BatchNorm2D(outChannels),
                new Activation(fn)
            ]);
        }
        else
        {
            return new SequentialBlock([
                new Conv2D(outChannels, inChannels, 1, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1)),
            new Activation(fn),
            new Conv2D(outChannels, outChannels, 1, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1)),
            new Activation(fn)
            ]);
        }
    }

    private INetworkModule BuildRecursiveUNet(
        int depth,
        int maxDepth,
        int inChannels,
        int baseFeatures,
        int targetHeight,
        int targetWidth,
        ActivationFunction fn,
        bool useNormalization
    )
    {
        int level = maxDepth - depth;
        int prevFeatures = baseFeatures * (1 << (level - 1));
        int currentFeatures = baseFeatures * (1 << level);
        int nextFeatures = baseFeatures * (1 << (level + 1));

        if (depth == 0)
        {
            // Bottleneck block: just a double conv 
            // (is this the issue? it outputs baseFeatures not nextFeatures)
            // (this results in the TransposeConv2D having the wrong number of inChannelsPerGroup)
            return ConvBlock(inChannels, currentFeatures, fn, useNormalization);
        }

        // Encoder block
        var encoder = ConvBlock(inChannels, currentFeatures, fn, useNormalization);

        // Downsample
        var downsample = new MaxPool2D(size: 2, stride: 2, padding: 0);

        // Compute expected size after downsampling
        int downHeight = (targetHeight + 1) / 2;
        int downWidth = (targetWidth + 1) / 2;

        // Recursive descent
        var deeperUNet = BuildRecursiveUNet(
            depth - 1,
            maxDepth,
            inChannels: currentFeatures,
            baseFeatures: baseFeatures,
            targetHeight: downHeight,
            targetWidth: downWidth,
            fn: fn,
            useNormalization
        );

        // Upsample
        var upsample = new TransposeConv2D(
            outChannels: currentFeatures,
            inChannelsPerGroup: nextFeatures, // Inchannels per group is wrong. Why!?
            groups: 1,
            kernel: (2, 2),
            stride: (2, 2),
            dilation: (1, 1),
            inputPadding: (0, 0, 0, 0),
            outputPadding: (0, 0, 0, 0)
        );

        // Compose main path
        var mainPath = new SequentialBlock([
            downsample,
            deeperUNet,
            upsample
        ]);

        // Do the residual concatenation with CenterCrop to align skip path with main path
        var skipPath = new Center2D(rows: targetHeight, columns: targetWidth);
        var concat = new ResidualConcat(axis: ^3, mainPath, skipPath); // Concat along the Channels axis

        // Decoder block (after concat, so channel count doubles)
        var decoder = ConvBlock(currentFeatures * 2, currentFeatures, fn, useNormalization);

        return new SequentialBlock([encoder, concat, decoder]);
    }

    public INetworkModule Make(BuildSettings settings)
    {
        var ishape = new TensorShape(Math.Max(1, settings.InputChannels), Math.Max(0, settings.ImageHeight), Math.Max(0, settings.ImageWidth));

        // Build the full UNet recursively
        var unet = new SequentialBlock([
            BuildRecursiveUNet(
                depth: Math.Max(1, settings.Depth),
                maxDepth: Math.Max(1, settings.Depth),
                inChannels: Math.Max(1, settings.InputChannels),
                baseFeatures: Math.Max(1, settings.BaseFeatureCount),
                targetHeight: Math.Max(0, settings.ImageHeight),
                targetWidth: Math.Max(0, settings.ImageWidth),
                fn: settings.Activation,
                useNormalization: settings.UseNormalization
            ),
            // Final 1x1 conv to map features to output RGB image
            new Conv2D(
                outChannels: Math.Max(1, settings.OutputChannels),
                inChannelsPerGroup: Math.Max(1, settings.BaseFeatureCount),
                groups: 1,
                kernel: (1, 1),
                stride: (1, 1),
                dilation: (1, 1),
                padding: (0, 0, 0, 0)
            )
        ]);

        return new ArchitectureBlock(
            name: "UNet",
            inputShape: ishape,
            rootModule: unet
        );
    }
}
