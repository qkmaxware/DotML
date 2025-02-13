using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test.Layers;

[TestClass]
public class FullyConnectedLayerTest {
    [TestMethod]
    public void TestInput5Output3() {
        var layer = new FullyConnectedLayer(input_size: 5, neurons: 3);

        var X = Matrix<double>.FromFlattened(5, 1, [
            1.9631073474884033,
            0.6613199710845947,
            1.3225351572036743,
            -0.7096538543701172,
            0.4306403398513794
        ]);
        var W = Matrix<double>.FromFlattened(3, 5, [
            -0.05311301350593567,
            0.1283712387084961,
            -0.363670289516449,
            -0.429769366979599,
            0.043726563453674316,
            0.31933659315109253,
            0.05960935354232788,
            0.12985146045684814,
            0.11157244443893433,
            -0.31467416882514954,
            0.19784271717071533,
            0.015455901622772217,
            -0.043171048164367676,
            0.10961490869522095,
            -0.44419926404953003
        ]);
        Assert.AreEqual(layer.Weights.Shape, W.Shape);
        layer.Weights = W;
        var B = Vec<double>.Wrap([
            0.37308549880981445,
            -0.10959410667419434,
            -0.15381550788879395
        ]);
        Assert.AreEqual(layer.Biases.Dimensionality, B.Dimensionality);
        layer.Biases = B;

        var Y_truth = Matrix<double>.FromFlattened(3, 1, [
            0.19656458497047424,
            0.5137627124786377,
            -0.08138172328472137
        ]);
        var Y_projected = layer.EvaluateSync(new BatchedFeatureSet<double>(new FeatureSet<double>(X)))[0,0];
        Assert.AreEqual(Y_truth.Rows, Y_projected.Rows);
        Assert.AreEqual(Y_truth.Columns, Y_projected.Columns);

        using (var writer = new StreamWriter("FullyConnectedLayerTest.TestInput5Output3.YProjected.txt")) {
            writer.Write(Y_projected);
        }
        using (var writer = new StreamWriter("FullyConnectedLayerTest.TestInput5Output3.YTruth.txt")) {
            writer.Write(Y_truth);
        }
        foreach (var (projected, truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Feedforward failed, projected value does not equal truth. Compare YProjected.txt to YTruth.txt.");
        }

        var dY = Matrix<double>.FromFlattened(3, 1, [
            -0.863419771194458,
            0.3522203266620636,
            0.17849189043045044
        ]);
        var dX_truth = Matrix<double>.FromFlattened(5, 1, [
            0.19364899396896362,
            -0.08708388358354568,
            0.35203075408935547,
            0.4299348294734955,
            -0.2278749793767929
        ]);
        var dW_truth = Matrix<double>.FromFlattened(3, 5, [
            -1.6949857473373413,
            -0.5709967613220215,
            -1.141903042793274,
            0.6127291917800903,
            -0.37182337045669556,
            0.6914463043212891,
            0.23293033242225647,
            0.465823769569397,
            -0.24995450675487518,
            0.15168027579784393,
            0.3503987491130829,
            0.11804024875164032,
            0.23606179654598236,
            -0.12666745483875275,
            0.07686580717563629
        ]);
        var dB_truth = Vec<double>.Wrap([
            -0.863419771194458,
            0.3522203266620636,
            0.17849189043045044
        ]);

        var backprop_returns = layer.Backpropagate(new BackpropagationArgs(
            layer: -1, 
            input: new BatchedFeatureSet<double>(new FeatureSet<double>(X)),
            output: new BatchedFeatureSet<double>(new FeatureSet<double>(Y_truth)),
            error: new BatchedFeatureSet<double>(new FeatureSet<double>(dY))
        ));
        Assert.IsInstanceOfType<FullyConnectedLayer.Gradients>(backprop_returns.Gradients);
        var gradients = (FullyConnectedLayer.Gradients)backprop_returns.Gradients; 

        var dX_projected = backprop_returns.dX[0,0];
        var dW_projected = gradients.WeightGradients;
        var dB_projected = gradients.BiasGradients;

        using (var writer = new StreamWriter("FullyConnectedLayerTest.TestInput5Output3.dWProjected.txt")) {
            writer.Write(dW_projected);
        }
        using (var writer = new StreamWriter("FullyConnectedLayerTest.TestInput5Output3.dWTruth.txt")) {
            writer.Write(dW_truth);
        }
        using (var writer = new StreamWriter("FullyConnectedLayerTest.TestInput5Output3.dBProjected.txt")) {
            writer.Write(dB_projected);
        }
        using (var writer = new StreamWriter("FullyConnectedLayerTest.TestInput5Output3.dBTruth.txt")) {
            writer.Write(dB_truth);
        }
        using (var writer = new StreamWriter("FullyConnectedLayerTest.TestInput5Output3.dXProjected.txt")) {
            writer.Write(dX_projected);
        }
        using (var writer = new StreamWriter("FullyConnectedLayerTest.TestInput5Output3.dXTruth.txt")) {
            writer.Write(dX_truth);
        }

        foreach (var (projected, truth) in dW_projected.Zip(dW_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(W) value does not equal truth. Compare dXProjected.txt to dXTruth.txt.");
        }
        foreach (var (projected, truth) in dB_projected.Zip(dB_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(B) value does not equal truth. Compare dXProjected.txt to dXTruth.txt.");
        }
        foreach (var (projected, truth) in dX_projected.Zip(dX_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(X) value does not equal truth. Compare dXProjected.txt to dXTruth.txt.");
        }
    }
}