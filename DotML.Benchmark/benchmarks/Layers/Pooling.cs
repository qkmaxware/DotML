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
    [Params(32, 64, 128, 227)]
    public int LENGTH {get; set;}

    [Benchmark]
    public void TestMaxPoolFF() {
        var input = Enumerable.Range(0, IMG_CHANNELS).Select(x => new Matrix<double>(LENGTH, LENGTH)).ToArray();
        var layer = new LocalMaxPoolingLayer(new Shape3D(IMG_CHANNELS, LENGTH, LENGTH), KERNEL);
        
        var _result = layer.EvaluateSync(new FeatureSet<double>(input));
    }

    [Benchmark]
    public void TestAvgPoolFF() {
        var input = Enumerable.Range(0, IMG_CHANNELS).Select(x => new Matrix<double>(LENGTH, LENGTH)).ToArray();
        var layer = new LocalAvgPoolingLayer(new Shape3D(IMG_CHANNELS, LENGTH, LENGTH), KERNEL);
        
        var _result = layer.EvaluateSync(new FeatureSet<double>(input));
    }

}