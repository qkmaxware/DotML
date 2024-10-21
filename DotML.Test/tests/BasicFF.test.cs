using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class BasicFFTest {
    [TestMethod]
    public void TestNOT() {
        var network = new ClassicalFeedforwardNetwork(
            new NeuronLayer(1, 1) { 
                Weights             = new double[][]{ new double[]{ -1 } },
                Biases              = new double[] { 1 },
                ActivationFunctions = new ActivationFunction[]{}
            } // Output layer
        );
        var outputs = network.PredictSync(new double[]{
            1.0
        });
        Assert.AreEqual(false, outputs[0] > 0.5);
        outputs = network.PredictSync(new double[]{
            0.0,
        });
        Assert.AreEqual(true, outputs[0] > 0.5);
    }

    [TestMethod]
    public void TestAND() {
        var network = new ClassicalFeedforwardNetwork(
            new NeuronLayer(2, 1) { 
                Weights             = new double[][]{ new double[]{ 0.4, 0.4 } },
                ActivationFunctions = new ActivationFunction[]{}
            } // Output layer
        );
        var outputs = network.PredictSync(new double[]{
            1.0,
            1.0
        });
        Assert.AreEqual(true, outputs[0] > 0.5);
        outputs = network.PredictSync(new double[]{
            1.0,
            0.0
        });
        Assert.AreEqual(false, outputs[0] > 0.5);
        outputs = network.PredictSync(new double[]{
            0.0,
            1.0
        });
        Assert.AreEqual(false, outputs[0] > 0.5);
        outputs = network.PredictSync(new double[]{
            0.0,
            0.0
        });
        Assert.AreEqual(false, outputs[0] > 0.5);
    }

    [TestMethod]
    public void TestOR() {
        var network = new ClassicalFeedforwardNetwork(
            new NeuronLayer(2, 1) { 
                Weights             = new double[][]{ new double[]{ 0.6, 0.6 } },
                ActivationFunctions = new ActivationFunction[]{}
            } // Output layer
        );
        var outputs = network.PredictSync(new double[]{
            1.0,
            1.0
        });
        Assert.AreEqual(true, outputs[0] > 0.5);
        outputs = network.PredictSync(new double[]{
            1.0,
            0.0
        });
        Assert.AreEqual(true, outputs[0] > 0.5);
        outputs = network.PredictSync(new double[]{
            0.0,
            1.0
        });
        Assert.AreEqual(true, outputs[0] > 0.5);
        outputs = network.PredictSync(new double[]{
            0.0,
            0.0
        });
        Assert.AreEqual(false, outputs[0] > 0.5);
    }

    [TestMethod]
    public void TestXOR() {
        var test_inputs = new double[][] {
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        };
        var expected_outputs = new bool[]{
            false,
            true,
            true,
            false
        };

        // Weights/Biases copied from video: https://www.youtube.com/watch?v=kNPGXgzxoHw
        var network = new ClassicalFeedforwardNetwork(
            new NeuronLayer(2, 2) {
                Weights = [[20, 20], [-20, -20]],
                Biases = [-10, 30],
                ActivationFunctions = [Sigmoid.Instance, Sigmoid.Instance]
            },
            new NeuronLayer(2, 1) {
                Weights = [[20, 20]],
                Biases = [-30],
                ActivationFunctions = [Sigmoid.Instance]
            }
        );
        Console.WriteLine(network.ToJson());

        var actual_outputs = new double[test_inputs.Length];
        for (var i = 0; i < test_inputs.Length; i++) {
            actual_outputs[i] = network.PredictSync(test_inputs[i])[0];
        }

        for (var i = 0; i < test_inputs.Length; i++) {
            Console.WriteLine("For input [" + string.Join(',', test_inputs[i]) + "]; expected [" + (expected_outputs[i] ? 1.0 : 0.0) + "] got output [" + actual_outputs[i] + "]");
        }

        for (var i = 0; i < test_inputs.Length; i++) {
            Assert.AreEqual(expected_outputs[i], actual_outputs[i] > 0.5, "Incorrect answer for input [" + string.Join(',', test_inputs[i]) + "]");
        }
    }

    [TestMethod]
    public void TestIteratingTrainingData() {
        var trainingData = new TrainingSet(new List<TrainingPair> {
            new TrainingPair { Input = new Vec<double>(1.0, 1.0), Output = new Vec<double>(0.0) },
            new TrainingPair { Input = new Vec<double>(1.0, 0.0), Output = new Vec<double>(1.0) },
            new TrainingPair { Input = new Vec<double>(0.0, 1.0), Output = new Vec<double>(1.0) },
            new TrainingPair { Input = new Vec<double>(0.0, 0.0), Output = new Vec<double>(0.0) }
        });

        var sequential = trainingData.SampleSequentially();
        var index = 0;
        while (sequential.MoveNext()) {
            Assert.AreEqual(trainingData[index], sequential.Current);
            index ++; 
        }
        Assert.AreEqual(trainingData.Size, index);

        var random = trainingData.SampleRandomly();
        index = 0;
        var set = new Dictionary<TrainingPair, int>();
        foreach (var pair in trainingData) {
            set[pair] = 0;
        }
        while (random.MoveNext()) {
            if (set.ContainsKey(random.Current)) {
                set[random.Current] += 1;
            } else {
                set[random.Current] = 1;
            }
            index++;
        }
        Assert.AreEqual(false, set.Where(x => x.Value > 1).Any(), "Some training pairs were used more than once");
        Assert.AreEqual(false, set.Where(x => x.Value < 1).Any(), "Some training pairs were not used at all");
        Assert.AreEqual(trainingData.Size, index);
    }

    [TestMethod]
    public void TestXORUsingTraining() {
        // Create the network, make sure it was made correctly
        var network = new ClassicalFeedforwardNetwork(2,2,1);
        Assert.AreEqual(2, network.LayerCount);
        Assert.AreEqual(2, network.GetLayer(0).NeuronCount);
        Assert.AreEqual(1, network.GetLayer(1).NeuronCount);

        network.ForeachNeuron((ILayerWithNeurons layer, INeuron neuron) => neuron.ActivationFunction = HyperbolicTangent.Instance);
        network.ForeachNeuron((ILayerWithNeurons layer, INeuron neuron) => {
            Assert.AreEqual(HyperbolicTangent.Instance, neuron.ActivationFunction);
        });

        var trainingData = new List<TrainingPair> {
            new TrainingPair { Input = new Vec<double>(1.0, 1.0),   Output = new Vec<double>(-1.0) },
            new TrainingPair { Input = new Vec<double>(1.0, -1.0),  Output = new Vec<double>(1.0)  },
            new TrainingPair { Input = new Vec<double>(-1.0, 1.0),  Output = new Vec<double>(1.0)  },
            new TrainingPair { Input = new Vec<double>(-1.0, -1.0), Output = new Vec<double>(-1.0) }
        };
        var trainingSet = new TrainingSet(trainingData);

        var iterations = 500;
        var learning_rate = 0.1;
        var trainer = new EnumerableBackpropagationTrainer<ClassicalFeedforwardNetwork>();
        trainer.Epochs = iterations;
        trainer.LearningRate = learning_rate;
        trainer.Train(
            network, 
            trainingSet.SampleRandomly(), 
            trainingSet.SampleSequentially()
        );

        Console.WriteLine($"After Training ({iterations} iterations):");
        var measured = new Vec<double>[trainingData.Count];
        for (var i = 0; i < trainingData.Count; i++) {
            var output = network.PredictSync(trainingData[i].Input);
            Console.WriteLine($"For {trainingData[i].Input} = {trainingData[i].Output} got {output}");
            measured[i] = output;
        }
        Console.WriteLine(network.ToJson());

        for (var i = 0; i < trainingData.Count; i++) {
            var pair = trainingData[i];
            var output = measured[i];
            for (var j = 0; j < output.Dimensionality; j++) {
                Assert.AreEqual(pair.Output[j] > 0.0 ? true : false, output[j] > 0.0, $"Result mismatch. For input {pair.Input}, expected {pair.Output[j]}, got {output[j]} in output {output}");
            }
            //Console.WriteLine($"Correct result for {pair.Input} := {pair.Output} == {output}");
        }
    }
}