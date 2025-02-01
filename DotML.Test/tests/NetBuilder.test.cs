using System.Numerics;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class NetBuilderTest {

    [TestMethod]
    public void TestParsingSimple() {
var network_221 = 
@"FROM scratch 
INPUT 1 2 1
NAME '221-Network'

ADD dense neurons=2
ADD dense neurons=1";

        var lang = new NetBuild();
        var network = lang.ParseAndBuild(network_221);

        Assert.AreEqual(new Shape3D(1, 2, 1), network.InputShape);
        Assert.AreEqual("221-Network", network.Name);
        Assert.AreEqual(2, network.LayerCount);
        Assert.IsInstanceOfType<FullyConnectedLayer>(network.GetLayer(0));
        Assert.AreEqual(2, ((FullyConnectedLayer)network.GetLayer(0)).NeuronCount);
        Assert.IsInstanceOfType<FullyConnectedLayer>(network.GetLayer(1));
        Assert.AreEqual(1, ((FullyConnectedLayer)network.GetLayer(1)).NeuronCount);
    }
}