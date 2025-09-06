using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkTransposeConvolutionalLayer {

    public int KERNEL = 3;
    public int IMG_CHANNELS = 64;
    [Params(16, 32, 64, 128)]
    public int DIM_LENGTH {get; set;}

    private BatchedFeatureSet<float> input;
    private BatchedFeatureSet<float> output;
    private TransposeConvolutionLayer layer;

    [GlobalSetup]
    public void Setup() {
        var layer = new TransposeConvolutionLayer(
            input_size: new Shape3D(IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH), 
            padding: Padding.Same, 
            expansion: Expansion.Same,
            strideX: 1, strideY: 1,
            filters: ConvolutionFilter.Make(IMG_CHANNELS, IMG_CHANNELS, KERNEL)
        );
        var input = new FeatureSet<float>(layer.InputShape);
        var output = new FeatureSet<float>(layer.OutputShape);

        this.input = new BatchedFeatureSet<float>(input);
        this.output = new BatchedFeatureSet<float>(output);
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