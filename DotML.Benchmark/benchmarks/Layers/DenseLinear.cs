using BenchmarkDotNet.Attributes;
using CommandLine;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkDenseLinearLayer {

    [Params(3)]
    public int IMG_CHANNELS = 3;
    [Params(10)]
    public int OUT_CLASSES = 10;
    [Params(32, 64, 128, 256, 512, 1024)]
    public int DIM_LENGTH {get; set;}

    private BatchedFeatureSet<double> input;
    private BatchedFeatureSet<double> output;
    private DenseLinearLayer layer;

    [GlobalSetup]
    public void Setup() {
        var input = Enumerable.Range(0, IMG_CHANNELS).Select(x => new Matrix<double>(DIM_LENGTH, DIM_LENGTH)).ToArray();
        this.input = new BatchedFeatureSet<double>(new FeatureSet<double>(input));

        var output = Matrix<double>.Column(new Vec<double>(OUT_CLASSES));
        this.output = new BatchedFeatureSet<double>(new FeatureSet<double>(output));

        var layer = new DenseLinearLayer(input_size: new Shape3D(3, DIM_LENGTH, DIM_LENGTH).Count, OUT_CLASSES);
        this.layer = layer;
    }

    [Benchmark]
    public void Forward() {
        var _result = layer.EvaluateSync(input);
    }

    [Benchmark]
    public void Backward() {
        var gradients = layer.Backpropagate(new Network.Training.BackpropagationArgs(
            layer: -1,
            input: input,
            output: output,
            error: output
        ));
    }

}