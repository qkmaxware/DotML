using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class SerializationFFTest {
    [TestMethod]
    public void JsonSerialization() {
        var before = new ClassicalFeedforwardNetwork(
            new Layer(2, 2) {
                Weights = [[20, 20], [-20, -20]],
                Biases = [-10, 30],
                ActivationFunctions = [Sigmoid.Instance, Sigmoid.Instance]
            },
            new Layer(2, 1) {
                Weights = [[20, 20]],
                Biases = [-30],
                ActivationFunctions = [Sigmoid.Instance]
            }
        );

        var json = before.ToJson();

        var after = ClassicalFeedforwardNetwork.FromJson(json);

        Console.WriteLine(json);
        Console.WriteLine(after.ToJson());

        // Verify shape
        Assert.AreEqual(before.LayerCount, after.LayerCount);
        Assert.AreEqual(before.NeuronCount, after.NeuronCount);
        for (var layerIndex = 0; layerIndex < before.LayerCount; layerIndex++) {
            Assert.AreEqual(before.GetLayer(layerIndex).NeuronCount, after.GetLayer(layerIndex).NeuronCount);
        }

        // Verify properties
        var before_neurons = before.Layers.SelectMany(layer => layer.Neurons);
        var after_neurons = after.Layers.SelectMany(layer => layer.Neurons);
        var pairs = before_neurons.Zip(after_neurons);
        foreach (var pair in pairs) {
            // Equal activation
            Assert.AreEqual(pair.First.ActivationFunction?.GetType(), pair.Second.ActivationFunction?.GetType());

            // Equal biases
            Assert.AreEqual(pair.First.Bias,  pair.Second.Bias);
            
            // Equal weights
            if (pair.First.Weights is not null && pair.Second.Weights is not null) {
                Assert.AreEqual(pair.First.Weights.Length, pair.Second.Weights.Length);
                for (var j = 0; j < pair.First.Weights.Length; j++) {
                    Assert.AreEqual(pair.First.Weights[j], pair.Second.Weights[j]);
                }
            }
        }
    }
}