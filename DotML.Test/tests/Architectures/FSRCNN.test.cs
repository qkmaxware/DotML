using DotML.Network;
using DotML.Network.Templates;
using DotML.Network.Training;

namespace DotML.Test.Architectures;

[TestClass]
public class FSRCNNTest {
    [TestMethod]
    public void Build4x()
    {
        var factory = new FSRCNNFactory();

        // Build network
        var settings = new FSRCNNFactory.BuildSettings(upscalingFactor: 4);
        var network = (ArchitectureBlock)factory.Make(settings);
        Assert.IsNotNull(network.RequiredInputShape);

        // Test forward pass
        var input = Tensor<float>.Random(network.RequiredInputShape.Value, Distributions.Uniform<float>(0f, 1f));
        var ctx = new EvaluationContext(EvaluationMode.Training);
        var result = network.Forward(input, ctx);

        Assert.AreEqual(settings.UpscalingFactor * settings.ImgHeight, result.Shape[^2]);
        Assert.AreEqual(settings.UpscalingFactor * settings.ImgWidth, result.Shape[^1]);

        // Test backward pass
        var gradient = Tensor<float>.Random(result.Shape, Distributions.Uniform<float>(0f, 1f));
        var grads = network.Backward(gradient, ctx);
    }
}