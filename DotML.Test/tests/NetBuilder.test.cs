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

ADD DenseLinear input_size=2 neurons=2
ADD DenseLinear input_size=2 neurons=1";

        var lang = new NetbuildSerializer();
        var network = (SequentialBlock)lang.DeserializeModule(network_221);

        Assert.AreEqual("221-Network", network.Alias);
        Assert.AreEqual(2, network.LayerCount);
        Assert.IsInstanceOfType<DenseLinear>(network.GetLayer(0));
        Assert.AreEqual(2, ((DenseLinear)network.GetLayer(0)).Neurons);
        Assert.IsInstanceOfType<DenseLinear>(network.GetLayer(1));
        Assert.AreEqual(1, ((DenseLinear)network.GetLayer(1)).Neurons);
    }
}