using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class CompareDenseLinear
{
    #region Input Shape
    [Params(8)]
    public int BATCHES;
    [Params(1)]
    public int CHANNELS;
    [Params(4096)]
    public int ROWS;
    [Params(1)]
    public int COLUMNS;
    #endregion

    #region Output Shape
    [Params(100)]
    public int OUTPUT_CLASSES;
    #endregion

    private TensorShape ishape;
    private Tensor<float> inputTensor;
    private Tensor<float> outputTensor;

    [GlobalSetup]
    public void Setup()
    {
        var generator = new Random();
        ishape = new TensorShape(BATCHES, CHANNELS, ROWS, COLUMNS);
        var shape4 = new Shape4D(BATCHES, CHANNELS, ROWS, COLUMNS);

        updated = new DenseLinear(ROWS, OUTPUT_CLASSES);

        inputTensor = Tensor<float>.Generate(ishape, () => (float)generator.NextDouble());
        outputTensor = Tensor<float>.Defaults(updated.ForwardShape(ishape));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        // No need to do anything
    }

    DenseLinear updated;

    [Benchmark()]
    public void ForwardUpdated()
    {
        var ctx = new EvaluationContext();
        var output = updated.Forward(inputTensor, ctx);
    }

    [Benchmark()]
    public void BackwardUpdated()
    {
        var ctx = new EvaluationContext();
        var output = updated.Forward(inputTensor, ctx);
        updated.Backward(outputTensor, ctx);
    }
}