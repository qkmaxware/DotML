using DotML.Network;

namespace DotML.Examples.InfiniteShakespeare;

public class ArchitectureFactory
{  

    public int WindowSize = 16;
    public int EmbeddingSize = 3;
    public int VocabSize = 128;
	
    public int FeatureChannels = 32;        // Representation capacity
    public int HeadHiddenSize = 128;        // control classification flexibility
    public int TemporalKernel = 3;          // Control context modelling: try 3,5,7 etc
    public float DropoutRate = 0.1f;
    public ActivationFunction Function = ActivationFunctions.LeakyReLU;

    /// <summary>
    /// Input (after reshape): [B, 1, windowSize, embeddingDim]
    /// Output: [B, vocabSize]  (logits, no softmax)
    /// </summary>
    /// <returns></returns>
    public INetworkModule Make()
    {
        if (TemporalKernel % 2 == 0)
            throw new ArgumentException(nameof(TemporalKernel), "TemporalKernel must be an odd number");

        // Temporal convolution over embedding window
        var temporalBlock = new SequentialBlock([
            new Conv2D(
                outChannels: FeatureChannels, 
                inChannelsPerGroup: 1,
                groups: 1,
                kernel: new Size2D(width: EmbeddingSize, height: TemporalKernel), // n-gram
                stride: new Stride2D(x: 1, y: 1),
                dilation: new Dilation2D(x: 1, y: 1),
                padding: new Padding2D(left: 0, right: 0, top: TemporalKernel / 2, bottom: TemporalKernel / 2)
            ),
            new Activation(Function),
            new LayerNorm(FeatureChannels, WindowSize, 1)
        ]);

        // Main learning path
        var mainLearningPath = new SequentialBlock([
            new Conv2D(
                outChannels: FeatureChannels,
                inChannelsPerGroup: FeatureChannels, 
                groups: 1,
                kernel: new Size2D(width: 1, height: TemporalKernel),
                stride: new Stride2D(x: 1, y: 1),
                dilation: new Dilation2D(x: 1, y: 1),
                padding: new Padding2D(left: 0, right: 0, top: TemporalKernel / 2, bottom: TemporalKernel / 2)
            ),
            new Activation(Function),
            new Conv2D(
                outChannels: FeatureChannels,
                inChannelsPerGroup: FeatureChannels, 
                groups: 1,
                kernel: new Size2D(width: 1, height: TemporalKernel),
                stride: new Stride2D(x: 1, y: 1),
                dilation: new Dilation2D(x: 1, y: 1),
                padding: new Padding2D(left: 0, right: 0, top: TemporalKernel / 2, bottom:  TemporalKernel / 2)
            )
        ]);

        // Main residual path
        var residualLearningPath = new SequentialBlock([
            new Conv2D(
                outChannels: FeatureChannels,
                inChannelsPerGroup: FeatureChannels, 
                groups: 1,
                kernel: new Size2D(width: 1, height: 1),
                stride: new Stride2D(x: 1, y: 1),
                dilation: new Dilation2D(x: 1, y: 1),
                padding: new Padding2D(left: 0, right: 0, top: 0, bottom: 0)
            )
        ]);

        // Residual merge
        var learningBlock = new SequentialBlock([
            new ResidualAdd(mainLearningPath, residualLearningPath),
            new Activation(Function)
        ]);
        
        // Global temporal pooling
        var temporalPoolingBlock = new SequentialBlock([
            new AvgPool2D(
                width: 1,
                height: WindowSize,
                strideX: 1,
                strideY: 1,
                paddingX: 0,
                paddingY: 0
            )
        ]);

        // Prediction head
        var predictionBlock = new SequentialBlock([
            new Flatten(Flatten.FlatteningMode.CollapseNonBatch),
            new DenseLinear(input_size: FeatureChannels, neurons: HeadHiddenSize),
            new Activation(Function),
            new Dropout(DropoutRate),
            new DenseLinear(input_size: HeadHiddenSize, neurons: VocabSize)
        ]);

        return new ArchitectureBlock(
            name: "CNN Next-Word Prediction",
            inputShape: new DotML.Shape(1, 1, WindowSize, EmbeddingSize), // batch is placeholder
            rootModule: new SequentialBlock([
                temporalBlock,
                learningBlock,
                temporalPoolingBlock,
                predictionBlock
            ])
        );
    }

}