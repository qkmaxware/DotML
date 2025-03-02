using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkDenseLinearLayer {

    [Params(3)]
    public int IMG_CHANNELS = 3;
    [Params(10)]
    public int OUT_CLASSES = 10;
    [Params(32, 64, 128, 227)]
    public int LENGTH {get; set;}

    [Benchmark]
    public void TestConvolutionFF() {
        var input = Enumerable.Range(0, IMG_CHANNELS).Select(x => new Matrix<double>(LENGTH, LENGTH)).ToArray();
        var layer = new DenseLinearLayer(input_size: new Shape3D(3, LENGTH, LENGTH).Count, OUT_CLASSES);
        
        var _result = layer.EvaluateSync(new FeatureSet<double>(input));
    }

}