using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class CompareTransposeConv2D
{
    #region Input Shape
    [Params(256)]
    public int CHANNELS;
    [Params(27)]
    public int ROWS;
    [Params(27)]
    public int COLUMNS;
    #endregion

    #region Output Shape
    [Params(384)]
    public int OUT_CHANNELS;
    #endregion

    private TensorShape ishape;
    private FeatureSet<float> inputMatrix;
    private FeatureSet<float> outputMatrix;
    private Tensor<float> inputTensor;
    private Tensor<float> outputTensor;

    [GlobalSetup]
    public void Setup()
    {
        var generator = new Random();
        ishape = new TensorShape(CHANNELS, ROWS, COLUMNS);

        var updated_layer = new TransposeConv2D(
            outChannels: OUT_CHANNELS,
            inChannelsPerGroup: CHANNELS,
            groups: 1,
            kernel: (3, 3),
            stride: (1, 1),
            dilation: (1, 1),
            inputPadding: (0, 0, 0, 0),
            outputPadding: (0, 0, 0, 0)
        );
        this.updated = updated_layer;
        var old_layer = new TransposeConvolutionLayer(
            new Shape3D(ishape.Length(0), ishape.Length(1), ishape.Length(2)),
            Padding.Valid,
            Expansion.Same,
            1, 1,
            ConvolutionFilter.Make(filters: OUT_CHANNELS, kernels_per_filter: CHANNELS, kernel_size: 3)
        );
        this.old = old_layer;

        if (updated_layer.ForwardShape(ishape).Equals(old.OutputShape))
            throw new Exception($"These are not equivalent layers {updated_layer.Weights.Shape} vs {old_layer.FilterShape}");

        inputMatrix = new FeatureSet<float>(old.InputShape);
        inputTensor = Tensor<float>.Generate(ishape, () => (float)generator.NextDouble());
        outputMatrix = new FeatureSet<float>(old.OutputShape);
        outputTensor = Tensor<float>.Generate(old.OutputShape, () => (float)generator.NextDouble());
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        // No need to do anything
    }

    FeedforwardNetworkLayer old;
    INetworkModule updated;

    [Benchmark()]
    public void ForwardOld()
    {
        var output = old.EvaluateSync(inputMatrix);
    }
    
    [Benchmark()]
    public void ForwardUpdated()
    {
        var output = updated.Forward(inputTensor);
    }

    [Benchmark()]
    public void BackwardOld()
    {
        var output = old.EvaluateSync(inputMatrix);
        old.Backpropagate(new Network.Training.BackpropagationArgs(
            layer: -1,
            input: new BatchedFeatureSet<float>(inputMatrix),
            output: new BatchedFeatureSet<float>(output),
            error: new BatchedFeatureSet<float>(outputMatrix)
        ));
    }

    [Benchmark()]
    public void BackwardUpdated()
    {
        var ctx = new EvaluationContext();
        var output = updated.Forward(inputTensor, ctx);
        updated.Backward(outputTensor, ctx);
    }
}