using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkActivationLayer {

    [Params(3)]
    public int IMG_CHANNELS = 3;
    [Params(10)]
    public int OUT_CLASSES = 10;
    [Params(32, 64, 128, 227)]
    public int LENGTH {get; set;}

    [Benchmark]
    public void TestReLUFF() {
        var input = Enumerable.Range(0, IMG_CHANNELS).Select(x => new Matrix<double>(LENGTH, LENGTH)).ToArray();
        var layer = new ActivationLayer(new Shape3D(IMG_CHANNELS, LENGTH, LENGTH), ActivationFunctions.ReLU);
        
        var _result = layer.EvaluateSync(new FeatureSet<double>(input));
    }

    [Benchmark]
    public void TestSigmoidFF() {
        var input = Enumerable.Range(0, IMG_CHANNELS).Select(x => new Matrix<double>(LENGTH, LENGTH)).ToArray();
        var layer = new ActivationLayer(new Shape3D(IMG_CHANNELS, LENGTH, LENGTH), ActivationFunctions.Sigmoid);
        
        var _result = layer.EvaluateSync(new FeatureSet<double>(input));
    }

    [Benchmark]
    public void TestTanhFF() {
        var input = Enumerable.Range(0, IMG_CHANNELS).Select(x => new Matrix<double>(LENGTH, LENGTH)).ToArray();
        var layer = new ActivationLayer(new Shape3D(IMG_CHANNELS, LENGTH, LENGTH), ActivationFunctions.Tanh);
        
        var _result = layer.EvaluateSync(new FeatureSet<double>(input));
    }

}