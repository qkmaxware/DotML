using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test.Layers;

[TestClass]
public class LocalMaxPoolingLayerTest {
    [TestMethod]
    public void TestLocalMaxPooling() {
        var layer = new LocalMaxPoolingLayer(new Shape3D(1, 4, 4), 2, 2);
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
            {20, 30},
            {112, 37}
        });
        Assert.AreEqual(result.Rows, output.Rows);
        Assert.AreEqual(result.Columns, output.Columns);
        for (var r = 0; r < result.Rows; r++) {
            for (var c = 0; c < result.Columns; c++) {
                Assert.AreEqual(result[r, c], output[r, c], $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
            }
        }
    }

    [TestMethod]
    public void TestStride1Padding0Kernel3() {
        var layer = new LocalMaxPoolingLayer(new Shape3D(1, 5, 5), size: 3, stride: 1);

        var X = Matrix<double>.FromFlattened(5, 5, [
            0.45132818818092346,
            -1.5761557817459106,
            -0.6721860766410828,
            -2.0042014122009277,
            -1.0609326362609863,
            -1.678432822227478,
            0.056944191455841064,
            1.0239132642745972,
            1.9709233045578003,
            0.7276046276092529,
            1.160701870918274,
            0.03253338485956192,
            -0.3749595284461975,
            -0.6730366349220276,
            -0.07208921015262604,
            1.8418018817901611,
            -1.1823792457580566,
            1.186455488204956,
            -0.880447268486023,
            -0.631264865398407,
            -1.4471193552017212,
            1.9196826219558716,
            -0.6389657855033875,
            0.6240558624267578,
            1.5673837661743164
        ]);

        var Y_truth = Matrix<double>.FromFlattened(3, 3, [
            1.160701870918274,
            1.9709233045578003,
            1.9709233045578003,
            1.8418018817901611,
            1.9709233045578003,
            1.9709233045578003,
            1.9196826219558716,
            1.9196826219558716,
            1.5673837661743164
        ]);
        var Y_projected = layer.EvaluateSync(new BatchedFeatureSet<double>(new FeatureSet<double>(X)))[0,0];
        Assert.AreEqual(Y_truth.Rows, Y_projected.Rows);
        Assert.AreEqual(Y_truth.Columns, Y_projected.Columns);

        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.YProjected.txt")) {
            writer.Write(Y_projected);
        }
        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.YTruth.txt")) {
            writer.Write(Y_truth);
        }
        foreach (var (projected, truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Feedforward failed, projected value does not equal truth. Compare YProjected.txt to YTruth.txt.");
        }

        var dY = Matrix<double>.FromFlattened(3, 3, [
            -0.31870266795158386,
            -0.391875684261322,
            0.10528147220611572,
            -0.46503812074661255,
            1.715201735496521,
            -1.2883495092391968,
            2.3749876022338867,
            -0.5080739259719849,
            1.6750047206878662
        ]);
        var dX_truth = Matrix<double>.FromFlattened(5, 5, [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.14025795459747314,
            0.0,
            -0.31870266795158386,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.46503812074661255,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.8669136762619019,
            0.0,
            0.0,
            1.6750047206878662
        ]);

        var backprop_returns = layer.Backpropagate(new BackpropagationArgs(
            layer: -1,
            input: new BatchedFeatureSet<double>(new FeatureSet<double>(X)),
            output: new BatchedFeatureSet<double>(new FeatureSet<double>(Y_truth)),
            error: new BatchedFeatureSet<double>(new FeatureSet<double>(dY))
        ));
        var dX_projected = backprop_returns.dX[0,0];
        Assert.IsNull(backprop_returns.Gradients);
        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.dXProjected.txt")) {
            writer.Write(dX_projected);
        }
        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.dXTruth.txt")) {
            writer.Write(dX_truth);
        }

        foreach (var (projected, truth) in dX_projected.Zip(dX_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(X) value does not equal truth. Compare dXProjected.txt to dXTruth.txt.");
        }
    }
}