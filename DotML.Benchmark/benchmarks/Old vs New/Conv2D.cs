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

    private Shape ishape;
    private Tensor<float> inputTensor;
    private Tensor<float> outputTensor;

    [GlobalSetup]
    public void Setup()
    {
        var generator = new Random();
        ishape = new Shape(BATCHES, CHANNELS, ROWS, COLUMNS);
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

        inputTensor = Tensor<float>.Generate(ishape, () => (float)generator.NextDouble());
        outputTensor = Tensor<float>.Generate(updated.ForwardShape(ishape), () => (float)generator.NextDouble());
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        // No need to do anything
    }

    INetworkModule updated;

    
    [Benchmark()]
    public void ForwardUpdated()
    {
        var output = updated.Forward(inputTensor);
    }

    [Benchmark()]
    public void BackwardUpdated()
    {
        var ctx = new EvaluationContext();
        var output = updated.Forward(inputTensor, ctx);
        updated.Backward(outputTensor, ctx);
    }
}