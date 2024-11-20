using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class ConvolutionalFFTest {
    [TestMethod]
    public void TestConvolutionalLayerValidPadding() {
        var kernels = new Matrix<double>[]{
            new double[,] {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1}
            }
        };
        var layer = new ConvolutionLayer(Padding.Valid, new ConvolutionFilter(kernels));
        var input = new Matrix<double>[] {
            new double[,]{
                {1, 1, 1, 0, 0},
                {0, 1, 1, 1, 0},
                {0, 0, 1, 1, 1},
                {0, 0, 1, 1, 0},
                {0, 1, 1, 0, 0},
            },
        };
        var outputs = layer.EvaluateSync(input);
        Assert.AreEqual(1, outputs.Length);
        var output = outputs[0];

        Matrix<double> result = new double[,] {
            {4, 3, 4},
            {2, 4, 3},
            {2, 3, 4}
        };

        Assert.AreEqual(result.Rows, output.Rows);
        Assert.AreEqual(result.Columns, output.Columns);
        for (var r = 0; r < result.Rows; r++) {
            for (var c = 0; c < result.Columns; c++) {
                Assert.AreEqual(result[r, c], output[r, c], $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
            }
        }
    }

    [TestMethod]
    public void TestConvolutionalLayerSamePadding() {
        var kernels = new Matrix<double>[]{
            new double[,] {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1}
            }
        };
        var layer = new ConvolutionLayer(Padding.Same, new ConvolutionFilter(kernels));
        Matrix<double> input = new double[,]{
            {1, 1, 1, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 1, 1, 1},
            {0, 0, 1, 1, 0},
            {0, 1, 1, 0, 0},
        };
        var inputs = new Matrix<double>[] {
            input
        };
        var outputs = layer.EvaluateSync(inputs);
        Assert.AreEqual(1, outputs.Length);
        var output = outputs[0];

        Matrix<double> result = new double[,]{
            {2, 2, 3, 1, 1},
            {1, 4, 3, 4, 1},
            {1, 2, 4, 3, 3},
            {1, 2, 3, 4, 1},
            {0, 2, 2, 1, 1},
        };

        Assert.AreEqual(input.Rows, output.Rows);
        Assert.AreEqual(input.Columns, output.Columns);

        for (var r = 0; r < output.Rows; r++) {
            for (var c = 0; c < output.Columns; c++) {
                Assert.AreEqual(result[r, c], output[r, c], $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
            }
        }
    }

    [TestMethod]
    public void TestLocalMaxPooling() {
        var layer = new LocalMaxPoolingLayer(2, 2);
        Matrix<double> input = new double[,] {
            {12, 20, 30, 00},
            {08, 12, 02, 00},
            {34, 70, 37, 04},
            {112, 100, 25, 12}
        };

        var outputs = layer.EvaluateSync([input]);
        Assert.AreEqual(1, outputs.Length);
        var output = outputs[0];

        Matrix<double> result = new double[,] {
            {20, 30},
            {112, 37}
        };
        Assert.AreEqual(result.Rows, output.Rows);
        Assert.AreEqual(result.Columns, output.Columns);
        for (var r = 0; r < result.Rows; r++) {
            for (var c = 0; c < result.Columns; c++) {
                Assert.AreEqual(result[r, c], output[r, c], $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
            }
        }
    }

    [TestMethod]
    public void TestAvgMaxPooling() {
        var layer = new LocalAvgPoolingLayer(2, 2);
        Matrix<double> input = new double[,] {
            {12, 20, 30, 00},
            {08, 12, 02, 00},
            {34, 70, 37, 04},
            {112, 100, 25, 12}
        };

        var outputs = layer.EvaluateSync([input]);
        Assert.AreEqual(1, outputs.Length);
        var output = outputs[0];

        Matrix<double> result = new double[,] {
            {13, 8},
            {79, 19.5}
        };
        Assert.AreEqual(result.Rows, output.Rows);
        Assert.AreEqual(result.Columns, output.Columns);
        for (var r = 0; r < result.Rows; r++) {
            for (var c = 0; c < result.Columns; c++) {
                Assert.AreEqual(result[r, c], output[r, c], 0.01, $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
            }
        }
    }

    

}