using System;

namespace DotML.Network.Templates;

/// <summary>
/// Factory that can produce VGG Convolutional Neural Networks
/// </summary>
public class VGGFactory
: INetworkModuleFactory<VGGFactory.BuildSettings>
{

    public INetworkModule MakeDefault() => Make(BuildSettings.VGG16());

    public class BuildSettings
    {
        /// <summary>
        /// Width of the image
        /// </summary>
        public int ImageWidth {get; set;} = 224;
        /// <summary>
        /// Height of the image
        /// </summary>
        public int ImageHeight {get; set;} = 224;
        /// <summary>
        /// Number of image channels (usually RGB)
        /// </summary>
        public int ImageChannels {get; set;} = 3;

        /// <summary>
        /// The activation function to use, if null ReLU is used
        /// </summary>
        public ActivationFunction? ActivationFunction {get; set;} = ActivationFunctions.ReLU;

        /// <summary>
        /// The number of, and size of each convolutional block
        /// </summary>
        public IEnumerable<int> ConvolutionBlockSizes = [2, 2];
        /// <summary>
        /// The number of, and size of each of the hidden FC layers
        /// </summary>
        public IEnumerable<int> HiddenNeuronCounts = [4096, 4096];
        /// <summary>
        /// Total number of weight bearing layers in the final network
        /// </summary>
        public int WeightedLayerCount => ConvolutionBlockSizes.Sum() + HiddenNeuronCounts.Count() + 1; // Convs + FCs + logit output

        /// <summary>
        /// Number of base filters
        /// </summary>
        public int BaseFilters {get; set;} = 64;

        /// <summary>
        /// Number of output classes / neurons in the last dense linear layer
        /// </summary>
        public int OutputClasses {get; set;} = 10000;

        /// <summary>
        /// Flag to indicate if Batch Normalization should be used
        /// </summary>
        public bool UseBatchNorm {get; set;} = false;

        /// <summary>
        /// Amount of dropout to use between 0 and 1
        /// </summary>
        public float DropoutPercent {get; set;} = 0.0f;

        /// <summary>
        /// VGG-6 settings for lightweight small VGG
        /// </summary>
        public static BuildSettings VGG6() => new BuildSettings()
        {
            ConvolutionBlockSizes = [2, 2],
            BaseFilters = 32,
            HiddenNeuronCounts = [256],
            OutputClasses = 10
        };

        /// <summary>
        /// VGG-7 settings for lightweight small VGG
        /// </summary>
        public static BuildSettings VGG7() => new BuildSettings()
        {
            ConvolutionBlockSizes = [2, 2],
            BaseFilters = 32,
            HiddenNeuronCounts = [256, 128],
            OutputClasses = 10
        };

        /// <summary>
        /// VGG-11 settings
        /// </summary>
        public static BuildSettings VGG11() => new BuildSettings()
        {
            ConvolutionBlockSizes = [1, 1, 2, 2, 2],
            BaseFilters = 64,
            HiddenNeuronCounts = [4096, 4096],
            OutputClasses = 1000
        };

        /// <summary>
        /// VGG-16 settings
        /// </summary>
        public static BuildSettings VGG16() => new BuildSettings()
        {
            ConvolutionBlockSizes = [2, 2, 3, 3, 3],
            BaseFilters = 64,
            HiddenNeuronCounts = [4096, 4096],
            OutputClasses = 1000
        };

        /// <summary>
        /// VGG-19 settings
        /// </summary>
        public static BuildSettings VGG19() => new BuildSettings()
        {
            ConvolutionBlockSizes = [2, 2, 4, 4, 4],
            BaseFilters = 64,
            HiddenNeuronCounts = [4096, 4096],
            OutputClasses = 1000
        };
    }   

    public INetworkModule Make(BuildSettings settings)
    {
        var ishape = new TensorShape(settings.ImageChannels, settings.ImageHeight, settings.ImageWidth); // CHW
        var activation = settings.ActivationFunction ?? ActivationFunctions.ReLU;
        var dropout = Math.Clamp(settings.DropoutPercent, 0.0f, 1.0f);
        var useDropout = dropout > 0.0f;

        var builder = SequentialBlock.Begin(ishape);

        int blockIndex = 0;
        foreach (var blockSize in settings.ConvolutionBlockSizes)
        {
            var channels = Math.Min(settings.BaseFilters * (1 << blockIndex), 512);

            // Add all Conv for this block
            for (var i = 0; i < blockSize; i++) {
                builder = builder.Then((shape) => new Conv2D(
                    outChannels: channels,
                    inChannelsPerGroup: shape[NCHW.Channels],
                    groups: 1,
                    kernel: 3,
                    stride: 1,
                    dilation: 1,
                    padding: Padding.Same.ToPadding(kernel: 3, stride: 1, dilation: 1)
                ));
                builder = builder.ThenIf(settings.UseBatchNorm, (shape) => new BatchNorm2D(shape[NCHW.Channels]));
                builder = builder.Then((shape) => new Activation(activation));
            }

            // Add the pooling for this block
            builder = builder.Then((shape) => new MaxPool2D(
                size: 2, 
                stride: 2, 
                padding: 0
            ));

            blockIndex++;
        } 

        // Add the classifiers/logit output
        foreach (var neurons in settings.HiddenNeuronCounts)
        {
            builder = builder
            .Then((shape) => new DenseLinear(
                input_size: shape.LogicalElementCount(), // C * H * W
                neurons: neurons
            ))
            .WithActivation(activation)
            .ThenIf(useDropout, new Dropout(dropout));
        };
        builder = builder.Then((shape) => new DenseLinear(
            input_size: shape.LogicalElementCount(), // C * H * W
            neurons: settings.OutputClasses
        ));
        
        // Create the network
        return new ArchitectureBlock(
            name: $"VGG-{settings.WeightedLayerCount}",
            inputShape: ishape,
            rootModule: builder.Finalize()
        );
    }

}