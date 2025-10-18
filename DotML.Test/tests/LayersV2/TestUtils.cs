using DotML;
using DotML.Network;

namespace DotML.Test.Layers;

public static class LayerTester
{

    public static void ForwardOnly(IWeightsAndBiasNetworkModule layer, Tensor<float> w, Tensor<float> b, Tensor<float> x, Tensor<float> y)
    {
        // Init
        Assert.AreEqual(layer.Weights.Shape, w.Shape);
        layer.Weights = w;
        Assert.AreEqual(layer.Biases.Shape, b.Shape);
        layer.Biases = b;

        // Forward
        var context = new EvaluationContext(EvaluationMode.Training);
        var y_projected = layer.Forward(x, context);
        Assert.AreEqual(true, y_projected.Equals(y, 0.0001f));
    }

    public static void ForwardAndBack(IWeightsAndBiasNetworkModule layer, Tensor<float> w, Tensor<float> b, Tensor<float> x, Tensor<float> y, Tensor<float> dy, Tensor<float> dx, Tensor<float> dw, Tensor<float> db)
    {
        // Init
        Assert.AreEqual(layer.Weights.Shape, w.Shape);
        layer.Weights = w;
        Assert.AreEqual(layer.Biases.Shape, b.Shape);
        layer.Biases = b;

        // Forward
        var context = new EvaluationContext(EvaluationMode.Training);
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
}