using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class SerializationTest {
    [TestMethod]
    public void SvgSerialization() {
        //var network = MultilayerPerceptron.Make(ActivationFunctions.Sigmoid, 2, 2, 1);
        var network = ResNet.Make(ResNet.Version.V6, 3, 32, 32);
        
        var svg = new DotML.Network.SvgWriter();
        using var writer = new StreamWriter("SerializationTest.SvgSerialization.svg");
        svg.WriteTo(network, writer);
    }
}