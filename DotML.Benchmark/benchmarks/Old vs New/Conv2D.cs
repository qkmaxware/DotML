using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class CompareConv2D
{
    #region Input Shape
    [Params(8)]
    public int BATCHES;
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
    private BatchedFeatureSet<float> inputMatrix;
    private BatchedFeatureSet<float> outputMatrix;
    private Tensor<float> inputTensor;
    private Tensor<float> outputTensor;

    [GlobalSetup]
    public void Setup()
    {
        var generator = new Random();
        ishape = new TensorShape(BATCHES, CHANNELS, ROWS, COLUMNS);
        var shape4 = new Shape4D(BATCHES, CHANNELS, ROWS, COLUMNS);

        var updated_layer = new Conv2D(
            outChannels: OUT_CHANNELS,
            inChannelsPerGroup: CHANNELS,
            groups: 1,
            kernel: (3, 3),
            stride: (1, 1),
            dilation: (1, 1),
            padding: Padding.Same.ToTuple(kernel: (3, 3), stride: (1, 1), dilation: (1, 1))
        );
        this.updated = updated_layer;
        var old_layer = new ConvolutionLayer(
            new Shape3D(ishape.Length(1), ishape.Length(2), ishape.Length(3)),
            Padding.Same,
            ConvolutionFilter.Make(filters: OUT_CHANNELS, kernels_per_filter: CHANNELS, kernel_size: 3)
        );
        this.old = old_layer;

        if (!updated.ForwardShape(ishape).Slice(1..).Equals((TensorShape)old_layer.OutputShape))
            throw new Exception($"These are not equivalent layers {ishape} -> {updated.ForwardShape(ishape)} vs {old_layer.InputShape} -> {old_layer.OutputShape}");

        inputMatrix = new BatchedFeatureSet<float>(shape4);
        inputTensor = Tensor<float>.Generate(ishape, () => (float)generator.NextDouble());
        outputMatrix = new BatchedFeatureSet<float>(new Shape4D(BATCHES, old.OutputShape.Channels, old.OutputShape.Rows, old.OutputShape.Columns));
        outputTensor = Tensor<float>.Generate(updated.ForwardShape(ishape), () => (float)generator.NextDouble());
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
            input: (inputMatrix),
            output: (output),
            error: (outputMatrix)
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