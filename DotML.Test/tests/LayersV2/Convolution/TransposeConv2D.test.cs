using System.Text.Json;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Test.Layers.Convolution;

[TestClass]
public partial class TransposeConv2DTest
{

    private static void Test(TransposeConv2D layer, Tensor<float> w, Tensor<float> b, Tensor<float> x, Tensor<float> y, Tensor<float> dy, Tensor<float> dx, Tensor<float> dw, Tensor<float> db)
    {
        // Init
        Assert.AreEqual(layer.Weights.Shape, w.Shape);
        layer.Weights = w;
        Assert.AreEqual(layer.Biases.Shape, b.Shape);
        layer.Biases = b;

        // Forward
        var context = new EvaluationContext();
        var y_projected = layer.Forward(x, context);
        Assert.AreEqual(true, y_projected.Equals(y, 0.0001f));

        // Backwards
        var back = layer.Backward(dy, context);
        Assert.IsInstanceOfType<WeightAndBiasGradients>(back);
        WeightAndBiasGradients gradients = (WeightAndBiasGradients)back;

        Assert.AreEqual(true, gradients.dB.Equals(db, 0.001f));
        Assert.AreEqual(true, gradients.dW.Equals(dw, 0.001f));
        Assert.AreEqual(true, gradients.dX.Equals(dx, 0.001f));
    }

    public class GeneratedTensorSet
    {
        public double[][][][]? X { get; set; }
        public double[][][][]? Y { get; set; }
        public double[][][][]? W { get; set; }
        public double[]? B { get; set; }

        public double[][][][]? dX { get; set; }
        public double[][][][]? dY { get; set; }
        public double[][][][]? dW { get; set; }
        public double[]? db { get; set; }
    }

    [TestMethod]
    public void TestBatches1Channels1Filters1Kernel2Stride1Padding0()
    {
        var layer = new TransposeConv2D(
            outChannels: 1,
            inChannelsPerGroup: 1,
            groups: 1,
            kernel: (2, 2),
            stride: (1, 1),
            dilation: (1, 1),
            inputPadding: (0, 0, 0, 0),
            outputPadding: (0, 0, 0, 0)
        );

        var tensors = JsonSerializer.Deserialize<GeneratedTensorSet>(TestBatches1Channels1Filters1Kernel2Stride1Padding0_tensors)!;
        var w = Tensor<double>.FromJaggedArray(tensors.W!).ToFloat();
        var dw = Tensor<double>.FromJaggedArray(tensors.dW!).ToFloat();
        var b = Tensor<double>.FromJaggedArray(tensors.B!).ToFloat();
        var db = Tensor<double>.FromJaggedArray(tensors.db!).ToFloat();
        var x = Tensor<double>.FromJaggedArray(tensors.X!).ToFloat();
        var dx = Tensor<double>.FromJaggedArray(tensors.dX!).ToFloat();
        var y = Tensor<double>.FromJaggedArray(tensors.Y!).ToFloat();
        var dy = Tensor<double>.FromJaggedArray(tensors.dY!).ToFloat();

        Test(
            layer: layer,

            w: w,
            dw: dw,
            b: b,
            db: db,

            x: x,
            dx: dx,

            y: y,
            dy: dy
        );
    }

    [TestMethod]
    public void TestBatches1Channels3Filters2Kernel3Stride2Padding1()
    {
        var layer = new TransposeConv2D(
            outChannels: 2,
            inChannelsPerGroup: 3,
            groups: 1,
            kernel: (3, 3),
            stride: (2, 2),
            dilation: (1, 1),
            inputPadding: (1, 1, 1, 1),
            outputPadding: (0, 0, 0, 0)
        );

        var tensors = JsonSerializer.Deserialize<GeneratedTensorSet>(TestBatches1Channels3Filters2Kernel3Stride2Padding1_tensors)!;
        var w = Tensor<double>.FromJaggedArray(tensors.W!).ToFloat();
        var dw = Tensor<double>.FromJaggedArray(tensors.dW!).ToFloat();
        var b = Tensor<double>.FromJaggedArray(tensors.B!).ToFloat();
        var db = Tensor<double>.FromJaggedArray(tensors.db!).ToFloat();
        var x = Tensor<double>.FromJaggedArray(tensors.X!).ToFloat();
        var dx = Tensor<double>.FromJaggedArray(tensors.dX!).ToFloat();
        var y = Tensor<double>.FromJaggedArray(tensors.Y!).ToFloat();
        var dy = Tensor<double>.FromJaggedArray(tensors.dY!).ToFloat();

        Test(
            layer: layer,

            w: w,
            dw: dw,
            b: b,
            db: db,

            x: x,
            dx: dx,

            y: y,
            dy: dy
        );
    }

}