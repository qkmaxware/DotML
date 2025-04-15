using DotML.Network;
using DotML.Network.IO.Netbuild;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class NetBuildWriterTest {

    [TestMethod]
    // CIFAR10
    public void MakeLeNet3x32x32_10() {
        var network = LeNet.Make(LeNet.Version.V5, 10, 3, 32, 32, ReLU.Instance);

        using var writer = new StreamWriter("letnet.3x32x32.10.netbuild");
        var netbuild = new NetbuildLayerEncoder(writer);

        netbuild.Encode(network);
    }

    [TestMethod]
    // DIGITS
    public void MakeLeNet1x32x32_10() {
        var network = LeNet.Make(LeNet.Version.V5, 10, 1, 32, 32, ReLU.Instance);

        using var writer = new StreamWriter("letnet.1x32x32.10.netbuild");
        var netbuild = new NetbuildLayerEncoder(writer);

        netbuild.Encode(network);
    }

    [TestMethod]
    // LETTERS
    public void MakeLeNet1x32x32_28() {
        var network = LeNet.Make(LeNet.Version.V5, 28, 1, 32, 32, ReLU.Instance);

        using var writer = new StreamWriter("letnet.1x32x32.28.netbuild");
        var netbuild = new NetbuildLayerEncoder(writer);

        netbuild.Encode(network);
    }

    [TestMethod]
    // OCR
    public void MakeLeNet1x32x32_38() { 
        var network = LeNet.Make(LeNet.Version.V5, 38, 1, 32, 32, ReLU.Instance);

        using var writer = new StreamWriter("letnet.1x32x32.38.netbuild");
        var netbuild = new NetbuildLayerEncoder(writer);

        netbuild.Encode(network);
    }

    [TestMethod]
    // OCR
    public void MakeAlexNet4x224x224_10() { 
        var network = AlexNet.Make(AlexNet.Version.V1, 10, img_width: 224, img_height: 224, activation: ReLU.Instance);

        using var writer = new StreamWriter("alexnet.3x224x224.10.netbuild");
        var netbuild = new NetbuildLayerEncoder(writer);

        netbuild.Encode(network);
    }

    [TestMethod]
    // Texture Upscale
    public void MakeFSRCNN_32_96() {
        var network = FSRCNN.Make(FSRCNN.Version.V1, img_channels: 3, img_width: 32, img_height: 32, scaling: 3, activation: ReLU.Instance);

        using var writer = new StreamWriter("fsrcnn.32.96.netbuild");
        var netbuild = new NetbuildLayerEncoder(writer);

        netbuild.Encode(network);
    }

    [TestMethod]
    // Texture Upscale
    public void MakeESPCN_32_96() {
        var network = ESPCN.Make(ESPCN.Version.V1, img_channels: 3, img_width: 32, img_height: 32, scaling: 3, activation: ReLU.Instance);

        using var writer = new StreamWriter("espcn.32.96.netbuild");
        var netbuild = new NetbuildLayerEncoder(writer);

        netbuild.Encode(network);
    }
}