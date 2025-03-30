using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkConvolutionalLayer {

    [Params(3)]
    public int KERNEL = 3;
    [Params(3)]
    public int IMG_CHANNELS = 3;
    [Params(10)]
    public int OUT_CLASSES = 10;
    [Params(32, 64, 128, 256, 512, 1024)]
    public int DIM_LENGTH {get; set;}

    private BatchedFeatureSet<double> input;
    private BatchedFeatureSet<double> output;
    private ConvolutionLayer layer;

    [GlobalSetup]
    public void Setup() {
        var layer = new ConvolutionLayer(input_size: new Shape3D(3, DIM_LENGTH, DIM_LENGTH), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(1, OUT_CLASSES, KERNEL));
        var input = new FeatureSet<double>(layer.InputShape);
        var output = new FeatureSet<double>(layer.OutputShape);

        this.input = new BatchedFeatureSet<double>(input);
        this.output = new BatchedFeatureSet<double>(output);
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