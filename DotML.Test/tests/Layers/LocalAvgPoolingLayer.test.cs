using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test.Layers;

[TestClass]
public class LocalAvgPoolingLayerTest {
    [TestMethod]
    public void TestAvgMaxPooling() {
        var layer = new LocalAvgPoolingLayer(new Shape3D(1, 4, 4), 2, 2);
        Matrix<double> input = new Matrix<double>(new double[,] {
            {12, 20, 30, 00},
            {08, 12, 02, 00},
            {34, 70, 37, 04},
            {112, 100, 25, 12}
        });

        var outputs = layer.EvaluateSync(new FeatureSet<double>(input));
        Assert.AreEqual(1, outputs.Channels);
        var output = outputs[0];

        Matrix<double> result = new Matrix<double>(new double[,] {
            {13, 8},
            {79, 19.5}
        });
        Assert.AreEqual(result.Rows, output.Rows);
        Assert.AreEqual(result.Columns, output.Columns);
        for (var r = 0; r < result.Rows; r++) {
            for (var c = 0; c < result.Columns; c++) {
                Assert.AreEqual(result[r, c], output[r, c], 0.01, $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
            }
        }
    }

    [TestMethod]
    public void TestStride1Padding0Kernel3() {
        var layer = new LocalAvgPoolingLayer(new Shape3D(1, 5, 5), size: 3, stride: 1);

        var X = Matrix<double>.FromFlattened(5, 5, [
            -0.06675183027982712,
            0.7418843507766724,
            -0.07433906942605972,
            -0.03677411377429962,
            -1.0757157802581787,

            0.7527914643287659,
            -0.2763320207595825,
            1.245193362236023,
            0.9814322590827942,
            0.2918619215488434,

            0.9653529524803162,
            -1.670727252960205,
            0.5376826524734497,
            -0.33155393600463867,
            -0.3139996826648712,

            0.37308916449546814,
            0.42045533657073975,
            0.4851781725883484,
            0.36806729435920715,
            -0.025358738377690315,

            0.4659097194671631,
            1.7895983457565308,
            0.6633204221725464,
            -0.22848817706108093,
            0.6940374970436096
        ]);

        var Y_truth = Matrix<double>.FromFlattened(3, 3, [
            0.2394171804189682,
            0.12405179440975189,
            0.1359763890504837,
            0.31474265456199646,
            0.19548842310905457,
            0.3598337471485138,
            0.4477621614933014,
            0.22594809532165527,
            0.20543172955513
        ]);
        var Y_projected = layer.EvaluateSync(new BatchedFeatureSet<double>(new FeatureSet<double>(X)))[0,0];
        Assert.AreEqual(Y_truth.Rows, Y_projected.Rows);
        Assert.AreEqual(Y_truth.Columns, Y_projected.Columns);

        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.YProjected.txt")) {
            writer.Write(Y_projected);
        }
        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.YTruth.txt")) {
            writer.Write(Y_truth);
        }
        foreach (var (projected, truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Feedforward failed, projected value does not equal truth. Compare YProjected.txt to YTruth.txt.");
        }

        var dY = Matrix<double>.FromFlattened(3, 3, [
            -2.179100513458252,
            -0.08496836572885513,
            -1.8678022623062134,
            1.0132676362991333,
            -0.5630502700805664,
            -0.2114965319633484,
            -1.1144579648971558,
            -1.0145455598831177,
            -0.3372708261013031
        ]);
        var dX_truth = Matrix<double>.FromFlattened(5, 5, [
            -0.24212227761745453,
            -0.2515632212162018,
            -0.4590967893600464,
            -0.21697451174259186,
            -0.2075335830450058,
            -0.12953698635101318,
            -0.2015390694141388,
            -0.43257224559783936,
            -0.30303525924682617,
            -0.23103320598602295,
            -0.2533656358718872,
            -0.4380950331687927,
            -0.7066026926040649,
            -0.4532370865345001,
            -0.2685077488422394,
            -0.011243373155593872,
            -0.18653179705142975,
            -0.24750596284866333,
            -0.23626258969306946,
            -0.06097415089607239,
            -0.12382866442203522,
            -0.23655594885349274,
            -0.274030476808548,
            -0.15020182728767395,
            -0.037474535405635834
        ]);

        var backprop_returns = layer.Backpropagate(new BackpropagationArgs(
            layer: -1,
            input: new BatchedFeatureSet<double>(new FeatureSet<double>(X)),
            output: new BatchedFeatureSet<double>(new FeatureSet<double>(Y_truth)),
            error: new BatchedFeatureSet<double>(new FeatureSet<double>(dY))
        ));
        var dX_projected = backprop_returns.dX[0,0];
        Assert.IsNull(backprop_returns.Gradients);
        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.dXProjected.txt")) {
            writer.Write(dX_projected);
        }
        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.dXTruth.txt")) {
            writer.Write(dX_truth);
        }

        foreach (var (projected, truth) in dX_projected.Zip(dX_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(X) value does not equal truth. Compare dXProjected.txt to dXTruth.txt.");
        }
    }
}