using DotML.Network;
using DotML.Network.Templates;
using DotML.Network.Training;

namespace DotML.Test.Architectures;

[TestClass]
public class ResNetTest
{
    [TestMethod]
    public void BuildResNet18()
    {
        var factory = new ResNetFactory();

        // Test building
        var settings = ResNetFactory.BuildSettings.ResNet18();
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