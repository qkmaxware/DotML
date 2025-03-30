using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkActivationLayer {

    [Params(3)]
    public int IMG_CHANNELS = 3;
    [Params(10)]
    public int OUT_CLASSES = 10;
    [Params(32, 64, 128, 256, 512, 1024)]
    public int DIM_LENGTH {get; set;}

    private BatchedFeatureSet<double> input;
    private BatchedFeatureSet<double> output;

    [GlobalSetup]
    public void Setup() {
        var layer = new ActivationLayer(new Shape3D(IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH), ActivationFunctions.ReLU);
        var input = new FeatureSet<double>(layer.InputShape);
        var output = new FeatureSet<double>(layer.OutputShape);

        this.input = new BatchedFeatureSet<double>(input);
        this.output = new BatchedFeatureSet<double>(output);
    }

    [Benchmark]
    public void ForwardReLU() {
        var layer = new ActivationLayer(new Shape3D(IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH), ActivationFunctions.ReLU);
        
        var _result = layer.EvaluateSync(input);
    }
    [Benchmark]
    public void BackwardReLU() {
        var layer = new ActivationLayer(new Shape3D(IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH), ActivationFunctions.ReLU);
        
        var gradients = layer.Backpropagate(new Network.Training.BackpropagationArgs(
            layer: -1,
            input: input,
            output: output,
            error: output
        ));
    }

    [Benchmark]
    public void ForwardSigmoid() {
        var layer = new ActivationLayer(new Shape3D(IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH), ActivationFunctions.Sigmoid);
        
        var _result = layer.EvaluateSync(input);
    }
    [Benchmark]
    public void BackwardSigmoid() {
        var layer = new ActivationLayer(new Shape3D(IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH), ActivationFunctions.Sigmoid);
        
        var gradients = layer.Backpropagate(new Network.Training.BackpropagationArgs(
            layer: -1,
            input: input,
            output: output,
            error: output
        ));
    }

    [Benchmark]
    public void ForwardTanh() {
        var layer = new ActivationLayer(new Shape3D(IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH), ActivationFunctions.Tanh);
        
        var _result = layer.EvaluateSync(input);
    }
    [Benchmark]
    public void BackwardTanh() {
        var layer = new ActivationLayer(new Shape3D(IMG_CHANNELS, DIM_LENGTH, DIM_LENGTH), ActivationFunctions.Tanh);
        
        var gradients = layer.Backpropagate(new Network.Training.BackpropagationArgs(
            layer: -1,
            input: input,
            output: output,
            error: output
        ));
    }

}