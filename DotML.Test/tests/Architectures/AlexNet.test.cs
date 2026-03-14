using DotML.Network;
using DotML.Network.Templates;
using DotML.Network.Training;

namespace DotML.Test.Architectures;

[TestClass]
public class AlexNetTest
{
    [TestMethod]
    public void BuildAlexNet()
    {
        var factory = new AlexNetFactory();

        // Test building
        var settings = new AlexNetFactory.BuildSettings();
        var network = (ArchitectureBlock)factory.Make(settings);
        Assert.IsNotNull(network.RequiredInputShape);

        // Test forward pass
        var input = Tensor<float>.Random(network.RequiredInputShape.Value, Distributions.Uniform<float>(0f, 1f));
        var ctx = new EvaluationContext(EvaluationMode.Training);
        var result = network.Forward(input, ctx);

        // Test backward pass
        var gradient = Tensor<float>.Random(result.Shape, Distributions.Uniform<float>(0f, 1f));
        var grads = network.Backward(gradient, ctx);
    }

    [TestMethod]
    public void BuildAlexNetWithNormalization()
    {
        var factory = new AlexNetFactory();

        // Test building
        var settings = new AlexNetFactory.BuildSettings();
        settings.NormalizeLayers = true;
        var network = (ArchitectureBlock)factory.Make(settings);
        Assert.IsNotNull(network.RequiredInputShape);

        // Test forward pass
        var input = Tensor<float>.Random(network.RequiredInputShape.Value, Distributions.Uniform<float>(0f, 1f));
        var ctx = new EvaluationContext(EvaluationMode.Training);
        var result = network.Forward(input, ctx);

        // Test backward pass
        var gradient = Tensor<float>.Random(result.Shape, Distributions.Uniform<float>(0f, 1f));
        var grads = network.Backward(gradient, ctx);
    }
}