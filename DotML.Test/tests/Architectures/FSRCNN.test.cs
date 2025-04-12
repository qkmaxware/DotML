using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test.Architectures;

[TestClass]
public class FSRCNNTest {
    [TestMethod]
    public void TestScalingFactor() {
        var input_size = FSRCNN.IMG_HEIGHT;
        for (var i = 1; i <= 6; i++) {
            var network = FSRCNN.Make(FSRCNN.Version.V1, scaling: i);
            Assert.AreEqual(i * input_size, network.GetOutputLayer().OutputShape.Rows);
            Assert.AreEqual(i * input_size, network.GetOutputLayer().OutputShape.Columns);
        }
    }
}