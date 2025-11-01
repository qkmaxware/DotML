using DotML.Network;
using DotML.Network.Templates;
using DotML.Network.Training;

namespace DotML.Test.Architectures;

[TestClass]
public class LeNetTest
{
    [TestMethod]
    public void BuildLeNet()
    {
        var factory = new LeNetFactory();

        // Test building
        var settings = new LeNetFactory.BuildSettingsV5();
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
    public void Svg()
    {
        var factory = new LeNetFactory();

        // Test building
        var settings = new LeNetFactory.BuildSettingsV5();
        var network = (ArchitectureBlock)factory.Make(settings);

        var renderer = new DotML.Network.IO.SvgRenderer();
        using var stream = new StreamWriter("LeNetTest.Svg.svg");
        renderer.RenderToStream(network, stream);
    }
}