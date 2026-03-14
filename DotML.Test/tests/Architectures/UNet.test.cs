using DotML.Network;
using DotML.Network.Templates;
using DotML.Network.Training;

namespace DotML.Test.Architectures;

[TestClass]
public class UNetTest
{
    [TestMethod]
    public void Build()
    {
        UNetFactory factory = new UNetFactory();

        var build = new UNetFactory.BuildSettings();

        build.InputChannels = 3;
        build.ImageWidth = 256;
        build.ImageHeight = 256;

        build.OutputChannels = 3;

        build.BaseFeatureCount = 64;

        build.Activation = ActivationFunctions.ReLU;

        // Test we can create the network and that it has the expected input/output shape
        var network = (ArchitectureBlock)factory.Make(build);
        if (!network.RequiredInputShape.HasValue || !network.OutputShape.HasValue)
        {
            Assert.Fail("Network does not have defined input/output shape.");
        }
        var ishape = new Shape(build.InputChannels, build.ImageHeight, build.ImageWidth);
        Assert.AreEqual(ishape, network.RequiredInputShape.Value);
        var oshape = new Shape(build.OutputChannels, build.ImageHeight, build.ImageWidth); 
        Assert.AreEqual(oshape, network.OutputShape.Value.Slice(1..)); // Remove batch dim

        // Test that we can actually run the network
        var input = Tensor<float>.Random(ishape, Distributions.Uniform<float>(0.0, 1.0));
        var output = network.Forward(input);
    }
}