using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class CompareDenseLinear
{
    #region Input Shape
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
    private Matrix<float> inputMatrix;
    private Matrix<float> outputMatrix;
    private Tensor<float> inputTensor;
    private Tensor<float> outputTensor;

    [GlobalSetup]
    public void Setup()
    {
        var generator = new Random();
        ishape = new TensorShape(CHANNELS, ROWS, COLUMNS);

        old = new DotML.Network.DenseLinearLayer(ROWS, OUTPUT_CLASSES);
        updated = new DenseLinear(ROWS, OUTPUT_CLASSES);

        inputMatrix = Matrix<float>.Generate(ROWS, COLUMNS, () => (float)generator.NextDouble());
        inputTensor = Tensor<float>.Generate(ishape, () => (float)generator.NextDouble());
        outputMatrix = new Matrix<float>(OUTPUT_CLASSES, 1);
        outputTensor = Tensor<float>.Defaults(new TensorShape(1, OUTPUT_CLASSES, 1));
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
        var output = old.EvaluateSync(new FeatureSet<float>(inputMatrix));
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
        var output = old.EvaluateSync(new FeatureSet<float>(inputMatrix));
        old.Backpropagate(new Network.Training.BackpropagationArgs(
            layer: -1,
            input: new BatchedFeatureSet<float>(new FeatureSet<float>(inputMatrix)),
            output: new BatchedFeatureSet<float>(output),
            error: new BatchedFeatureSet<float>(new FeatureSet<float>(outputMatrix))
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