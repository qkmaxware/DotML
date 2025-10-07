using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class CompareAvgPool
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

        old = new LocalAvgPoolingLayer(new Shape3D(CHANNELS, ROWS, COLUMNS), size: 3, stride: 1, padding: 0);
        updated = new AvgPool2D(size: 3, stride: 1, padding: 0);

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