using DotML;
using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test.Layers;

[TestClass]
public class SoftmaxLayerTest {
    [TestMethod]
    public void TestSafetensors() {
        var layer = new SoftmaxLayer(5);
        var writer = new LayerSafetensorWriter(); 
        layer.Visit(writer, 0);
        var tensors = writer.ToSafetensors();
        Assert.AreEqual(0, tensors.Keys().Count(), "Layer should not have any tensors.");
    }

    [TestMethod]
    public void TestSoftmax() {
        var layer = new SoftmaxLayer(5);
        var X = Matrix<double>.FromFlattened(5, 1, [
            0.6108487248420715,
            0.9583313465118408,
            0.11642353981733322,
            -2.0144927501678467,
            0.5680017471313477
        ]).ToFloatSet();
        var Y_truth = Matrix<double>.FromFlattened(5, 1, [
            0.24655473232269287,
            0.34899815917015076,
            0.15037901699543,
            0.017854267731308937,
            0.23621372878551483
        ]).ToFloatSet();
        var Y_projected = layer.EvaluateSync(new BatchedFeatureSet<float>(new FeatureSet<float>(X)))[0,0];
        Assert.AreEqual(Y_truth.Shape, Y_projected.Shape);
        foreach (var (projected, truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(truth, projected, 0.001);
        }

        var dY_truth = Matrix<double>.FromFlattened(5, 1, [
            1.1221718788146973,
            1.2508395910263062,
            -0.005234954878687859,
            -0.3422098755836487,
            -0.3510672152042389
        ]).ToFloatSet();
        var dX_truth = Matrix<double>.FromFlattened(5, 1, [
            0.1229761615395546,
            0.21897751092910767,
            -0.09453253448009491,
            -0.017240142449736595,
            -0.2301809936761856
        ]).ToFloatSet();
        var dX_projected = layer.Backpropagate(new BackpropagationArgs(
            layer: -1,
            input: new BatchedFeatureSet<float>(new FeatureSet<float>(X)),
            output: new BatchedFeatureSet<float>(new FeatureSet<float>(Y_truth)),
            error: new BatchedFeatureSet<float>(new FeatureSet<float>(dY_truth)) 
        )).dX[0,0];

        Assert.AreEqual(dX_truth.Shape, dX_projected.Shape);
        foreach (var (projected, truth) in dX_projected.Zip(dX_truth)) {
            Assert.AreEqual(truth, projected, 0.001);
        }
    }
}