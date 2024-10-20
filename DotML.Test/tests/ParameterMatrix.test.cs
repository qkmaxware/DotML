using System.Text.Json;
using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class ParameterMatrixTest {

    [TestMethod]
    public void TestElementCount() {
        var matrix = new ParameterMatrix(
            new Dictionary<string, object[]> {
                {"epoch",           [100, 200, 300, 400, 500]},
                {"learning_rate",   [0.001, 0.01, 0.1]},
            }
        );
        Assert.AreEqual(5 * 3, matrix.ToArray().Length);

        matrix = new ParameterMatrix(
            new Dictionary<string, object[]> {
                {"epoch",           [100, 200, 300, 400, 500]},
                {"learning_rate",   [0.001, 0.01, 0.1]},
                {"momentum",        [0.8, 0.9]},
            }
        );
        Assert.AreEqual(5 * 3 * 2, matrix.ToArray().Length);
    }

    // Not an actual test method, just want to make sure the syntax I want compiles
    // Good example of how to "discover" the best parameters for a problem set using brute force
    public void TestCompilationOfGenerator() {
        
        const int InputSize = 72;
        const int OutputSize = 26;
        TrainingSet training = new TrainingSet();
        TrainingSet validation = new TrainingSet();
        var generator = new TrainedParameterizedNetworkGenerator<ClassicalFeedforwardNetwork>(
            new ParameterMatrix (
                ("hidden_size",     Enumerable.Range(OutputSize, InputSize).Cast<object>().ToArray()),
                ("epoch",           [500, 1000]),
                ("learning_rate",   [0.001, 0.01, 0.1]),
                ("momentum",        [0, 0.8, 0.9])
            ),
            training,
            validation,
            (param) => new ClassicalFeedforwardNetwork(InputSize, param.Get<int>("hidden_size"), OutputSize),
            (param) => new EnumerableBackpropagationTrainer<ClassicalFeedforwardNetwork> {
                // Hard-coded parameters
                EarlyStop = true,
                // Parameters fetched from the matrix
                Epochs = param.Get<int>("epoch"),
                LearningRate = param.Get<int>("learning_rate"),
                MomentumFactor = param.Get<double>("momentum"),
            }
        );

        // Train the networks and return them all
        var networks = generator.GenerateAndTrain().ToArray();
        
        // Evaluate these networks to get the best one...
        // Hmm this doesn't let us get the parameter set of the best one
        // I mean maybe if I can get the index of the best one then I can use the parameter matrix to get it's params
        var loss = networks.Select(net => NetworkEvaluationFunctions.MaxMeanSquaredError(net, validation)).ToList();
        var bestFitness = loss.Min();                                       // Get the network with the least loss
        var bestIndex = loss.IndexOf(bestFitness);                          // Get the index of the network 
        var best_params = generator.ParameterMatrix.ElementAt(bestIndex);   // Get the param set for the best network
        Console.WriteLine(JsonSerializer.Serialize(best_params));       
    }

    [TestMethod]
    public void TestSequence() {
        var matrix = new ParameterMatrix(
            ("epoch",           [100, 200, 300, 400, 500]),
            ("learning_rate",   [0.001, 0.01, 0.1]),
            ("momentum",        [0.8, 0.9])
        );
        var validations = new List<(int, double, double)> {
            (100, 0.001, 0.8),
            (100, 0.001, 0.9),
            (100, 0.01, 0.8),
            (100, 0.01, 0.9),
            (100, 0.1, 0.8),
            (100, 0.1, 0.9),

            (200, 0.001, 0.8),
            (200, 0.001, 0.9),
            (200, 0.01, 0.8),
            (200, 0.01, 0.9),
            (200, 0.1, 0.8),
            (200, 0.1, 0.9),

            (300, 0.001, 0.8),
            (300, 0.001, 0.9),
            (300, 0.01, 0.8),
            (300, 0.01, 0.9),
            (300, 0.1, 0.8),
            (300, 0.1, 0.9),

            (400, 0.001, 0.8),
            (400, 0.001, 0.9),
            (400, 0.01, 0.8),
            (400, 0.01, 0.9),
            (400, 0.1, 0.8),
            (400, 0.1, 0.9),

            (500, 0.001, 0.8),
            (500, 0.001, 0.9),
            (500, 0.01, 0.8),
            (500, 0.01, 0.9),
            (500, 0.1, 0.8),
            (500, 0.1, 0.9),
        };
        Assert.AreEqual(5 * 3 * 2, validations.Count);

        var validation_index = 0;
        foreach (var param_set in matrix) {
            var validation = validations[validation_index];
            Assert.AreEqual(3, param_set.Count);
            Assert.AreEqual(validation.Item1, param_set.Get<int>("epoch"));
            Assert.AreEqual(validation.Item2, param_set.Get<double>("learning_rate"));
            Assert.AreEqual(validation.Item3, param_set.Get<double>("momentum"));
            validation_index++;
        }
    }

}