using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class CompareDropout
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

        old = new DropoutLayer(new Shape3D(CHANNELS, ROWS, COLUMNS), 0.5f);
        updated = new Dropout(0.5f);

        inputMatrix = new FeatureSet<float>(old.InputShape);
        inputTensor = Tensor<float>.Defaults(ishape);
        outputMatrix = new FeatureSet<float>(old.OutputShape);
        outputTensor = Tensor<float>.Defaults(old.OutputShape);
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
        old.BeginTraining(); // Regenerate dropout mask
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
        old.BeginTraining(); // Regenerate dropout mask
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