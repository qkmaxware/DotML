using System;

namespace DotML.Network.Templates;

/// <summary>
/// Implementation of the SimpleNet CNN classifier network as described in: <see href="https://arxiv.org/abs/1608.06037"/>.
/// </summary>
public class SimpleNetFactory
: INetworkModuleFactory<SimpleNetFactory.BuildSettings>
{
    public enum ParameterCount
    {
        FiveMillion,
        EightMillion
    }
    public class BuildSettings
    {
        public int ImageChannels {get; set;} = 3;
        public int ImageHeight {get; set;} = 32;
        public int ImageWidth {get; set;} = 32;

        public int OutputClasses {get; set;} = 10;
        public ParameterCount Size {get; set;} = ParameterCount.FiveMillion;
        public float Scale {get; set;} = 1.0f;

        public ActivationFunction ActivationFunction {get; set;} = ActivationFunctions.ReLU;

        public bool UseBatchNorm {get; set;} = true;
    }

    private int FloorToInt(float value, int? min = null)
    {
        var floored = (int)Math.Floor(value);
        if (min.HasValue)
            return Math.Max(floored, min.Value);
        return floored;
    }

    public INetworkModule Make(BuildSettings settings)
    {   
        // NCHW order
        var ishape = new Shape(settings.ImageChannels, settings.ImageHeight, settings.ImageWidth);

        var activation = settings.ActivationFunction ?? ReLU.Instance;

        /*
            // Channels, Stride, Dropout, LayerType
            (64, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            ("p", 2, 0.0),
            (256, 1, 0.0),
            (256, 1, 0.0),
            (256, 1, 0.0),
            (512, 1, 0.0),
            ("p", 2, 0.0),
            (2048, 1, 0.0, "k1"),
            (256, 1, 0.0, "k1"),
            (256, 1, 0.0),
        */

        return new ArchitectureBlock(
            name: $"SimpleNet-Small-M2-{settings.Scale}",
            inputShape: ishape,
            rootModule: 
                SequentialBlock.Begin(ishape)
                // Block 1
                .Then((shape) => new Conv2D(
                    outChannels: FloorToInt(64, min: 1),
                    inChannelsPerGroup: shape[NCHW.Channels],
                    groups: 1,
                    kernel: (3, 3),
                    stride: (2, 2),
                    dilation: (1, 1),
                    padding: (1, 1, 1, 1)
                ))
                .ThenIf(settings.UseBatchNorm, (shape) => new BatchNorm2D(shape[NCHW.Channels]))
                .Then((shape) => new Activation(activation))
                .Finalize()
        );
    }

}