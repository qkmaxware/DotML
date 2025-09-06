using BenchmarkDotNet.Attributes;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkAlexNet {
    const int IMG_CHANNELS = 3;
    const int OUT_CLASSES = 10;

    [Params(227)]
    public int DimensionLength {get; set;}

    private FeedforwardNetwork network;

    [GlobalSetup]
    public void Setup() {
        network = AlexNet.Make(AlexNet.Version.V1, output_classes: OUT_CLASSES, img_channels: IMG_CHANNELS, img_width: DimensionLength, img_height: DimensionLength, ActivationFunctions.ReLU);
    }

    [GlobalCleanup]
    public void Cleanup() {
        network = null;
    }

    [Benchmark]
    public void Forward() {
        var input = Enumerable.Range(0, IMG_CHANNELS).Select(x => new Matrix<float>(DimensionLength, DimensionLength)).ToArray();

        network.PredictSync(new FeatureSet<float>(input));
    }
}