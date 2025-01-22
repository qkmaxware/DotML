using System.Numerics;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class NetBuilderTest {

    [TestMethod]
    public void TestParsingSimple() {
var file = 
@"FROM scratch INPUT 1 2 1

NAME example

ADD dense neurons=2
ADD dense neurons=1";

        var lang = new NetBuild();
        var network = lang.Parse(file);

        Assert.AreEqual(new Shape3D(1, 2, 1), network.InputShape);
        Assert.AreEqual("example", network.Name);
        Assert.AreEqual(3, network.LayerCount);
        Assert.IsInstanceOfType<FullyConnectedLayer>(network.GetLayer(0));
        Assert.AreEqual(2, ((FullyConnectedLayer)network.GetLayer(0)).NeuronCount);
        Assert.IsInstanceOfType<FullyConnectedLayer>(network.GetLayer(1));
        Assert.AreEqual(1, ((FullyConnectedLayer)network.GetLayer(1)).NeuronCount);
    }
}