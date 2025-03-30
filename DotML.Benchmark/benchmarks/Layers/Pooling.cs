using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkPoolingLayer {

    [Params(3)]
    public int KERNEL = 3;
    [Params(3)]
    public int IMG_CHANNELS = 3;
    [Params(10)]
    public int OUT_CLASSES = 10;
    [Params(32, 64, 128, 256, 512, 1024)]
    public int DIM_LENGTH {get; set;}

    private FeedforwardNetworkLayer max;
    private FeedforwardNetworkLayer avg;
    private BatchedFeatureSet<double> input;
    private BatchedFeatureSet<double> output;

    [GlobalSetup]
    public void Setup() {
        var max = new LocalMaxPoolingLayer(new Shape3D(IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH), KERNEL);
        var avg = new LocalAvgPoolingLayer(new Shape3D(IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH), KERNEL);

        var input = new FeatureSet<double>(max.InputShape);
        var output = new FeatureSet<double>(max.OutputShape);

        this.max = max;
        this.avg = avg;
        this.input = new BatchedFeatureSet<double>(input);
        this.output = new BatchedFeatureSet<double>(output);
    }

    [Benchmark]
    public void ForwardMaxPool() {
        var _result = max.EvaluateSync(input);
    }
    [Benchmark]
    public void BackwardMaxPool() {
        var gradient = max.Backpropagate(new Network.Training.BackpropagationArgs(
            layer: -1,
            input: input,
            output: output,
            error: output
        ));
    }

    [Benchmark]
    public void ForwardAvgPool() {
        var _result = avg.EvaluateSync(input);
    }
    [Benchmark]
    public void BackwardAvgPool() {
        var gradient = avg.Backpropagate(new Network.Training.BackpropagationArgs(
            layer: -1,
            input: input,
            output: output,
            error: output
        ));
    }

}