using System;

namespace DotML.Network.Templates;

public class ResNetFactory
: INetworkModuleFactory<ResNetFactory.BuildSettings>
{
    public class BuildSettings
    {
        public int InputChannels { get; set; } = 3;
        public int InputHeight { get; set; } = 224;
        public int InputWidth { get; set; } = 224;
        public int NumClasses { get; set; } = 1000;
        public (int BlockCount, int OutChannels)[] Stages { get; set; } = new (int BlockCount, int OutChannels)[]
        {
            (3, 64), (4, 128), (6, 256), (3, 512)
        };
        public bool UseBottleneck { get; set; } = true;
        public ActivationFunction ActivationFunction { get; set; } = ActivationFunctions.ReLU;

        public static BuildSettings ResNet18(int numClasses = 1000)
        {
            return new BuildSettings
            {
                Stages = new[] { (2, 64), (2, 128), (2, 256), (2, 512) },
                UseBottleneck = false,
                NumClasses = numClasses
            };
        }

        public static BuildSettings ResNet34(int numClasses = 1000)
        {
            return new BuildSettings
            {
                Stages = new[] { (3, 64), (4, 128), (6, 256), (3, 512) },
                UseBottleneck = false,
                NumClasses = numClasses
            };
        }

        public static BuildSettings ResNet50(int numClasses = 1000)
        {
            return new BuildSettings
            {
                Stages = new[] { (3, 64), (4, 128), (6, 256), (3, 512) },
                UseBottleneck = true,
                NumClasses = numClasses
            };
        }

        public static BuildSettings ResNet101(int numClasses = 1000)
        {
            return new BuildSettings
            {
                Stages = new[] { (3, 64), (4, 128), (23, 256), (3, 512) },
                UseBottleneck = true,
                NumClasses = numClasses
            };
        }

        public static BuildSettings ResNet152(int numClasses = 1000)
        {
            return new BuildSettings
            {
                Stages = new[] { (3, 64), (8, 128), (36, 256), (3, 512) },
                UseBottleneck = true,
                NumClasses = numClasses
            };
        }
    }

    private static INetworkModule ResNetBottleneckBlockModule(ActivationFunction function, int inChannels, int outChannels, int stride, bool downsample)
    {
        var layers = new List<INetworkModule>();

        int bottleneckChannels = outChannels / 4;

        // 1x1 Conv (reduce channels)
        layers.Add(new Conv2D(
            inChannelsPerGroup: inChannels,
            outChannels: bottleneckChannels,
            groups: 1,
            kernel: (1, 1),
            stride: (1, 1),
            dilation: (1, 1),
            padding: (0, 0, 0, 0)
        ));
        layers.Add(new BatchNorm2D(bottleneckChannels));
        layers.Add(new Activation(function));

        // 3x3 Conv
        layers.Add(new Conv2D(
            inChannelsPerGroup: bottleneckChannels,
            outChannels: bottleneckChannels,
            groups: 1,
            kernel: (3, 3),
            stride: (stride, stride),
            dilation: (1, 1),
            padding: (1, 1, 1, 1)
        ));
        layers.Add(new BatchNorm2D(bottleneckChannels));
        layers.Add(new Activation(function));

        // 1x1 Conv (restore channels)
        layers.Add(new Conv2D(
            inChannelsPerGroup: bottleneckChannels,
            outChannels: outChannels,
            groups: 1,
            kernel: (1, 1),
            stride: (1, 1),
            dilation: (1, 1),
            padding: (0, 0, 0, 0)
        ));
        layers.Add(new BatchNorm2D(outChannels));

        // Shortcut connection
        INetworkModule? shortcut = null;
        if (downsample || inChannels != outChannels)
        {
            shortcut = new SequentialBlock(new INetworkModule[]
            {
                new Conv2D(
                    inChannelsPerGroup: inChannels,
                    outChannels: outChannels,
                    groups: 1,
                    kernel: (1, 1),
                    stride: (stride, stride),
                    dilation: (1, 1),
                    padding: (0, 0, 0, 0)
                ),
                new BatchNorm2D(outChannels)
            });
        }

        return new SequentialBlock([
            new ResidualAdd(
                new SequentialBlock(layers),
                shortcut
            ),
            new Activation(function)
        ]);
    }

    private static INetworkModule ResNetBasicBlockModule(ActivationFunction function, int inChannels, int outChannels, int stride, bool downsample)
    {
        var layers = new List<INetworkModule>();

        // 3x3 Conv
        layers.Add(new Conv2D(
            inChannelsPerGroup: inChannels,
            outChannels: outChannels,
            groups: 1,
            kernel: (3, 3),
            stride: (stride, stride),
            dilation: (1, 1),
            padding: (1, 1, 1, 1)
        ));
        layers.Add(new BatchNorm2D(outChannels));
        layers.Add(new Activation(function));

        // 3x3 Conv
        layers.Add(new Conv2D(
            inChannelsPerGroup: outChannels,
            outChannels: outChannels,
            groups: 1,
            kernel: (3, 3),
            stride: (1, 1),
            dilation: (1, 1),
            padding: (1, 1, 1, 1)
        ));
        layers.Add(new BatchNorm2D(outChannels));

        // Shortcut connection
        INetworkModule? shortcut = null;
        if (downsample || inChannels != outChannels)
        {
            shortcut = new SequentialBlock(new INetworkModule[]
            {
                new Conv2D(
                    inChannelsPerGroup: inChannels,
                    outChannels: outChannels,
                    groups: 1,
                    kernel: (1, 1),
                    stride: (stride, stride),
                    dilation: (1, 1),
                    padding: (0, 0, 0, 0)
                ),
                new BatchNorm2D(outChannels)
            });
        }

        return new SequentialBlock([
            new ResidualAdd(
                new SequentialBlock(layers),
                shortcut
            ),
            new Activation(function)
        ]);
    }

    public INetworkModule Make(BuildSettings settings)
    {
        var builder = new List<INetworkModule>();

        // Initial Conv + BN + ReLU + MaxPool
        builder.Add(new Conv2D(
            inChannelsPerGroup: settings.InputChannels,
            outChannels: 64,
            groups: 1,
            kernel: (7, 7),
            stride: (2, 2),
            dilation: (1, 1),
            padding: (3, 3, 3, 3)
        ));
        builder.Add(new BatchNorm2D(64));
        builder.Add(new Activation(settings.ActivationFunction));
        builder.Add(new MaxPool2D(size: 3, stride: 2, padding: 1));

        int inChannels = 64;
        for (int stage = 0; stage < settings.Stages.Length; stage++)
        {
            var (blocks, outChannels) = settings.Stages[stage];
            int stride = stage == 0 ? 1 : 2;

            for (int block = 0; block < blocks; block++)
            {
                bool downsample = block == 0 && stride != 1;
                if (settings.UseBottleneck)
                {
                    builder.Add(ResNetBottleneckBlockModule(
                        settings.ActivationFunction,
                        inChannels,
                        outChannels,
                        stride: block == 0 ? stride : 1,
                        downsample: downsample
                    ));
                }
                else
                {
                    builder.Add(ResNetBasicBlockModule(
                        settings.ActivationFunction,
                        inChannels,
                        outChannels,
                        stride: block == 0 ? stride : 1,
                        downsample: downsample
                    ));
                }
                inChannels = outChannels;
            }
        }

        builder.Add(new GlobalAvgPool2D());
        builder.Add(new DenseLinear(settings.Stages[^1].OutChannels, settings.NumClasses));

        return new SequentialBlock(builder);
    }
}