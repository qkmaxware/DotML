using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class PerformanceTest {

    const int IMG_WIDTH = 227;
    const int IMG_HEIGHT = 227;
    const int IMG_CHANNELS = 3;
    const int OUT_CLASSES = 3;

    [TestMethod]
    public void TestAlexNexConstruction() {
        ConvolutionalFeedforwardNetwork alexNet = new ConvolutionalFeedforwardNetwork(
            new ConvolutionLayer     (input_size: new Shape3D(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH), padding: Padding.Valid, stride: 4, filters: ConvolutionFilter.Make(96, 3, 11)),
            new ActivationLayer      (input_size: new Shape3D(96, 55, 55),  activation: HyperbolicTangent.Instance),
            new LocalMaxPoolingLayer (input_size: new Shape3D(96, 55, 55), size: 3, stride: 2),
            new ConvolutionLayer     (input_size: new Shape3D(96, 27, 27), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(256, 96, 5)),
            new ActivationLayer      (input_size: new Shape3D(256, 27, 27),  activation: HyperbolicTangent.Instance),
            new LocalMaxPoolingLayer (input_size: new Shape3D(256, 27, 27), size: 3, stride: 2),
            new ConvolutionLayer     (input_size: new Shape3D(256, 13, 13), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(384, 256, 3)),
            new ActivationLayer      (input_size: new Shape3D(384, 13, 13),  activation: HyperbolicTangent.Instance),
            new ConvolutionLayer     (input_size: new Shape3D(384, 13, 13), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(384, 384, 3)),
            new ActivationLayer      (input_size: new Shape3D(384, 13, 13),  activation: HyperbolicTangent.Instance),
            new ConvolutionLayer     (input_size: new Shape3D(384, 13, 13), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(256, 384, 3)),
            new ActivationLayer      (input_size: new Shape3D(256, 13, 13),  activation: HyperbolicTangent.Instance),
            new LocalMaxPoolingLayer (input_size: new Shape3D(256, 13, 13), size: 3, stride: 2),
            new FullyConnectedLayer  (input_size: 9216, neurons: 4096),
            new ActivationLayer      (input_size: new Shape3D(1, 4096, 1), activation: HyperbolicTangent.Instance),
            new FullyConnectedLayer  (input_size: 4096, neurons: 4096),
            new ActivationLayer      (input_size: new Shape3D(1, 4096, 1),  activation: HyperbolicTangent.Instance),
            new FullyConnectedLayer  (input_size: 4096, neurons: OUT_CLASSES),
            new ActivationLayer      (input_size: new Shape3D(1, OUT_CLASSES, 1),  activation: HyperbolicTangent.Instance),
            new SoftmaxLayer         (size: OUT_CLASSES)
        );

        using var writer = new StreamWriter("alexnet_20.md");
        writer.Write(alexNet.ToMarkdown());
    }

    [TestMethod]
    public void TestAlexNetFF() {
        Matrix<double>[] img = [new Matrix<double>(IMG_HEIGHT, IMG_WIDTH), new Matrix<double>(IMG_HEIGHT, IMG_WIDTH), new Matrix<double>(IMG_HEIGHT, IMG_WIDTH)];

        ConvolutionalFeedforwardNetwork alexNet = new ConvolutionalFeedforwardNetwork(
            new ConvolutionLayer     (input_size: new Shape3D(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH), padding: Padding.Valid, stride: 4, filters: ConvolutionFilter.Make(96, 3, 11)),
            new ActivationLayer      (input_size: new Shape3D(96, 55, 55), HyperbolicTangent.Instance),
            new LocalMaxPoolingLayer (input_size: new Shape3D(96, 55, 55), size: 3, stride: 2),
            new ConvolutionLayer     (input_size: new Shape3D(96, 27, 27), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(256, 96, 5)),
            new ActivationLayer      (input_size: new Shape3D(256, 27, 27), HyperbolicTangent.Instance),
            new LocalMaxPoolingLayer (input_size: new Shape3D(256, 27, 27), size: 3, stride: 2),
            new ConvolutionLayer     (input_size: new Shape3D(256, 13, 13), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(384, 256, 3)),
            new ActivationLayer      (input_size: new Shape3D(384, 13, 13), HyperbolicTangent.Instance),
            new ConvolutionLayer     (input_size: new Shape3D(384, 13, 13), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(384, 384, 3)),
            new ActivationLayer      (input_size: new Shape3D(384, 13, 13), HyperbolicTangent.Instance),
            new ConvolutionLayer     (input_size: new Shape3D(384, 13, 13), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(256, 384, 3)),
            new ActivationLayer      (input_size: new Shape3D(256, 13, 13), HyperbolicTangent.Instance),
            new LocalMaxPoolingLayer (input_size: new Shape3D(256, 13, 13), size: 3, stride: 2),
            new FullyConnectedLayer  (input_size: 9216, neurons: 4096),
            new ActivationLayer      (input_size: new Shape3D(1, 4096, 1), HyperbolicTangent.Instance),
            new FullyConnectedLayer  (input_size: 4096, neurons: 4096),
            new ActivationLayer      (input_size: new Shape3D(1, 4096, 1), HyperbolicTangent.Instance),
            new FullyConnectedLayer  (input_size: 4096, neurons: OUT_CLASSES),
            new ActivationLayer      (input_size: new Shape3D(1, OUT_CLASSES, 1), HyperbolicTangent.Instance),
            new SoftmaxLayer         (OUT_CLASSES)
        );

        var _result = alexNet.PredictSync(img);
    }

    [TestMethod]
    public void TestAlexNetConvoLayerFF() {
        Matrix<double>[] img = [new Matrix<double>(IMG_HEIGHT, IMG_WIDTH), new Matrix<double>(IMG_HEIGHT, IMG_WIDTH), new Matrix<double>(IMG_HEIGHT, IMG_WIDTH)];
        var layer = new ConvolutionLayer        (input_size: new Shape3D(3, IMG_HEIGHT, IMG_WIDTH), padding: Padding.Valid, stride: 4, filters: ConvolutionFilter.Make(96, 3, 11));

        var _result = layer.EvaluateSync(img);
    }

    [TestMethod]
    public void TestAlexNetPoolLayerFF() {
        var img = Enumerable.Range(0, 96).Select(x => new Matrix<double>(IMG_HEIGHT, IMG_WIDTH)).ToArray();
        var layer = new LocalMaxPoolingLayer    (new Shape3D(), size: 3, stride: 2);

        var _result = layer.EvaluateSync(img);
    }

    [TestMethod]
    public void TestAlexNetFullLayerFF() {
        var matrix = new Matrix<double>(9216,1);
        var layer = new FullyConnectedLayer     (9216, 4096);

        var _result = layer.EvaluateSync([matrix]);
    }

    [TestMethod]
    public void TestActivationLayer() {
        var matrix = new Matrix<double>(1024,1024);
        var layer = new ActivationLayer(matrix.Shape, HyperbolicTangent.Instance);

        var _result = layer.EvaluateSync([matrix]);
    }

}