using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class DeprecationTest {

    [TestMethod]
    public void ActivationLayerEquivalency() {
        var matrix = new double[,] {
            {1, 2},
            {3, 4}
        };
        var initializer = new HeInitialization();
        var layer = new FullyConnectedLayer(4, 4, HyperbolicTangent.Instance);
        layer.Initialize(initializer);
        var output1 = layer.EvaluateSync([matrix])[0];

        layer.ActivationFunction = Identity.Instance;
        var activation = new ActivationLayer(layer.OutputShape, HyperbolicTangent.Instance);
        var output2 = activation.EvaluateSync(layer.EvaluateSync([matrix]))[0];

        Assert.AreEqual(output1.Rows, output2.Rows);
        Assert.AreEqual(output1.Columns, output2.Columns);
        for (var r = 0; r < output1.Rows; r++) {
            for (var c = 0; c < output1.Columns; c++) {
                Assert.AreEqual(output1[r, c], output2[r, c], $"at {r},{c}");
            }
        }
    }

    [TestMethod]
    public void ActivationLayerTrainingEquivalency() {
        var input = new double[,] {
            {1, 2},
            {3, 4}
        };

        var tested_errors = new double[,] {
            {1},
            {2},
            {3},
            {4}
        };

        // Part 1, test in-place
        var initializer = new HeInitialization();
        var layer = new FullyConnectedLayer(4, 4, HyperbolicTangent.Instance);
        layer.Initialize(initializer);
        var outputs = layer.EvaluateSync([input]);

        var actions = new BatchedConvolutionalBackpropagationEnumerator<ConvolutionalFeedforwardNetwork>.BackpropagationActions(false, 0, 0);
        var args = new BatchedConvolutionalBackpropagationEnumerator<ConvolutionalFeedforwardNetwork>.BackpropagationArgs {
            Inputs = [input],
            Outputs = outputs,
            Errors = [tested_errors],
        };
        var results1 = layer.Visit(actions, args);

        // Part 2, test with removed activation layer
        layer.ActivationFunction = Identity.Instance;
        var activation = new ActivationLayer(layer.OutputShape, HyperbolicTangent.Instance);
        var results2_a = activation.Visit(actions, args);
        args.Errors = results2_a.Errors;
        var results2_b = layer.Visit(actions, args);

        // Assert results are identical
        Assert.AreEqual(1, results1.Errors.Length);
        Assert.AreEqual(results1.Errors.Length, results2_b.Errors.Length);
        Assert.AreEqual(results1.Errors[0].Rows, results2_b.Errors[0].Rows);
        Assert.AreEqual(results1.Errors[0].Columns, results2_b.Errors[0].Columns);
        for (var r = 0; r < results1.Errors[0].Rows; r++) {
            for (var c = 0; c < results1.Errors[0].Columns; c++) {
                Assert.AreEqual(results1.Errors[0][r, c], results2_b.Errors[0][r, c], $"at {r},{c}");
            }
        }

        Assert.IsNotNull(results1.Gradient);
        Assert.IsNotNull(results2_b.Gradient);
        var grad1 = (BatchedConvolutionalBackpropagationEnumerator<ConvolutionalFeedforwardNetwork>.FullyConnectedGradients)results1.Gradient;
        var grad2 = (BatchedConvolutionalBackpropagationEnumerator<ConvolutionalFeedforwardNetwork>.FullyConnectedGradients)results2_b.Gradient;

        Assert.AreEqual(grad1.BiasGradients.Dimensionality, grad2.BiasGradients.Dimensionality);
        for (var i = 0; i < grad1.BiasGradients.Dimensionality; i++) {
            Assert.AreEqual(grad1.BiasGradients[i], grad2.BiasGradients[i], $"at {i}");
        }
        Assert.AreEqual(grad1.WeightGradients.Rows, grad2.WeightGradients.Rows);
        Assert.AreEqual(grad1.WeightGradients.Columns, grad2.WeightGradients.Columns);
        for (var r = 0; r < grad1.WeightGradients.Rows; r++) {
            for (var c = 0; c < grad1.WeightGradients.Columns; c++) {
                Assert.AreEqual(grad1.WeightGradients[r, c], grad2.WeightGradients[r, c], $"at {r},{c}");
            }
        }

        //Assert.Fail(grad1.WeightGradients.ToString());
    }

}