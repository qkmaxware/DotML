using DotML;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Test.Layers.Softmax;

[TestClass]
public class SoftmaxOutputTest {
    [TestMethod]
    public void TestSafetensors() {
        var saver = new SafetensorSerializer();
        var loader = new SafetensorDeserializer();

        var layer = new SoftmaxOutput(^2);

        var tensors = saver.Serialize(layer);

        Assert.AreEqual(0, tensors.Count); // No tensors in activation layer

        loader.Deserialize(layer, tensors);
    }

    [TestMethod]
    public void TestSoftmax() {
        var layer = new SoftmaxOutput(^2);
        var X = Tensor<float>.FromFlattenedArray(new Shape(5, 1), [
            0.6108487248420715f,
            0.9583313465118408f,
            0.11642353981733322f,
            -2.0144927501678467f,
            0.5680017471313477f
        ]);
        var Y_truth = Tensor<float>.FromFlattenedArray(new Shape(5, 1), [
            0.24655473232269287f,
            0.34899815917015076f,
            0.15037901699543f,
            0.017854267731308937f,
            0.23621372878551483f
        ]);
        var Y_projected = layer.Forward(X);
        var equals = Y_projected.Equals(Y_truth, 0.001f);
        Assert.AreEqual(true, equals);
    }
}