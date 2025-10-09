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

        old = new DotML.Network.DenseLinearLayer(ROWS, OUTPUT_CLASSES);
        updated = new DenseLinear(ROWS, OUTPUT_CLASSES);

        inputMatrix = new BatchedFeatureSet<float>(shape4);
        inputTensor = Tensor<float>.Generate(ishape, () => (float)generator.NextDouble());
        outputMatrix = new BatchedFeatureSet<float>(new Shape4D(BATCHES, old.OutputShape.Channels, old.OutputShape.Rows, old.OutputShape.Columns));
        outputTensor = Tensor<float>.Defaults(updated.ForwardShape(ishape));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        // No need to do anything
    }

    DenseLinearLayer old;
    DenseLinear updated;

    [Benchmark()]
    public void ForwardOld()
    {
        var output = old.EvaluateSync(inputMatrix);
    }

    [Benchmark()]
    public void ForwardUpdated()
    {
        var ctx = new EvaluationContext();
        var output = updated.Forward(inputTensor, ctx);
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