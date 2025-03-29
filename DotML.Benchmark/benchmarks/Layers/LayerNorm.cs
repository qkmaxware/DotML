using BenchmarkDotNet.Attributes;
using DotML.Network;
using DotML.Network.Initialization;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkLayerNorm {

    [Params(3)]
    public int IMG_CHANNELS = 96;

    [Params(32, 64, 128, 256, 512, 1024)]
    public int DIM_LENGTH {get; set;}

    private BatchedFeatureSet<double> input;
    private LayerNorm layer;

    [GlobalSetup]
    public void Setup() {
        // Create random dataset
        input = new BatchedFeatureSet<double>(new Shape4D(1, IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH));
        Random rng = new Random();
        for (var b = 0; b < input.Batches; b++) { 
            for (var c = 0; c < input.Channels; c++) {
                var mtx = input[b, c];
                mtx.Apply((el) => rng.NextDouble());
            }
        }
        // Init layer
        layer = new LayerNorm(new Shape3D(IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH));
        layer.Initialize(Initializers.He);
    }

    [Benchmark]
    public void TestComputeMeansAndVariances() {
        layer.ComputeMeansAndVariances(input[0], out var mean, out var variance);
    }

    [Benchmark]
    public void TestForward() {
        var result = layer.EvaluateSync(input);
    }

    [Benchmark]
    public void TestBackwards() {
        var gradients = layer.Backpropagate(new Network.Training.BackpropagationArgs(
            layer: -1,
            input: input, 
            output: input,
            error: input
        ));
    }

}