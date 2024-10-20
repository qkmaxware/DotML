namespace DotML.Network.Training;

/// <summary>
/// Object that generates networks
/// </summary>
public interface INetworkGenerator<TNetwork> where TNetwork:INeuralNetwork {
    /// <summary>
    /// Generate a sequence of networks
    /// </summary>
    /// <returns>list of networks</returns>
    public IEnumerable<TNetwork> Generate();
}

/// <summary>
/// Behaviours for an object that both generates networks and trains them
/// </summary>
public interface ITrainedNetworkGenerator<TNetwork> : INetworkGenerator<TNetwork> where TNetwork:INeuralNetwork {
    /// <summary>
    /// Generate a sequence of fully trained networks
    /// </summary>
    /// <returns>list of networks all of which have undergone the training process</returns>
    IEnumerable<TNetwork> GenerateAndTrain();
}

public delegate TNetwork ParameterizedNetworkFactory<TNetwork>(ParameterSet @params) where TNetwork:INeuralNetwork;

/// <summary>
/// Class to generate a sequence of networks from a parameter matrix
/// </summary>
public class ParameterizedNetworkGenerator<TNetwork> : INetworkGenerator<TNetwork> where TNetwork:INeuralNetwork {
    private ParameterMatrix matrix;
    private ParameterizedNetworkFactory<TNetwork> factory;

    public ParameterizedNetworkGenerator(
        ParameterMatrix @params, 
        ParameterizedNetworkFactory<TNetwork> factory
    ) {
        this.matrix = @params;
        this.factory = factory;
    }

    /// <summary>
    /// Generate a sequence of networks
    /// </summary>
    /// <returns>list of networks</returns>
    public IEnumerable<TNetwork> Generate() {
        foreach (var set in matrix) {
            yield return factory(set);
        }
    }
}


public delegate ITrainer<TNetwork> ParameterizedTrainerFactory<TNetwork>(ParameterSet @params) where TNetwork:INeuralNetwork;

/// <summary>
/// Class to generate a sequence of networks from a parameter matrix and train them using a generated trainer
/// </summary>
public class TrainedParameterizedNetworkGenerator<TNetwork> : ITrainedNetworkGenerator<TNetwork> where TNetwork:INeuralNetwork {
    public ParameterMatrix ParameterMatrix {get; init;}
    private ParameterizedNetworkFactory<TNetwork> factory;
    private ParameterizedTrainerFactory<TNetwork> trainerFactory;

    public TrainingSet TrainingData {get; init;}
    public TrainingSet ValidationData {get; init;}

    public TrainedParameterizedNetworkGenerator(
        ParameterMatrix @params, 
        TrainingSet training,
        TrainingSet validation,
        ParameterizedNetworkFactory<TNetwork> netfactory, 
        ParameterizedTrainerFactory<TNetwork> trainerfactory
    ) {
        this.ParameterMatrix = @params;
        this.factory = netfactory;
        this.trainerFactory = trainerfactory;
        this.TrainingData = training;
        this.ValidationData = validation;
    }

    /// <summary>
    /// Generate a sequence of networks
    /// </summary>
    /// <returns>list of networks</returns>
    public IEnumerable<TNetwork> Generate() {
        // No training, just enumerate networks
        foreach (var set in ParameterMatrix) {
            yield return factory(set);
        }
    }

    /// <summary>
    /// Generate a sequence of networks and trainers
    /// </summary>
    /// <returns>list of network/trainer pairs</returns>
    public IEnumerable<(TNetwork, ITrainer<TNetwork>)> GenerateWithTrainers() {
        return this.ParameterMatrix.Select(p => (factory(p), trainerFactory(p)));
    }

    /// <summary>
    /// Generate a sequence of fully trained networks
    /// </summary>
    /// <returns>list of networks all of which have undergone the training process</returns>
    public IEnumerable<TNetwork> GenerateAndTrain() {
        var lst = this.ParameterMatrix.Select(p => (Network: factory(p), Trainer: trainerFactory(p))).ToArray();

        // Parallel training of all networks with all trainers
        Parallel.For(0, lst.Length, (index) => {
            var pair = lst[index];
            var trainer = pair.Trainer;
            var network = pair.Network;

            trainer.Train(network, TrainingData.SampleRandomly(), ValidationData.SampleSequentially());
        });

        // Enumerate all the trained networks
        foreach (var pair in lst) {
            yield return pair.Network;
        }
    }
}

// Usage
/*
const int InputSize = ...;
const int OutputSize = ...;
TrainingSet training = ...;
TrainingSet validation = ...;
var generator = new TrainedParameterizedNetworkGenerator<ClassicalFeedforwardNetwork>(
    new ParameterMatrix(
        ("hidden_size",     Enumerable.Range(26, 72).Cast<object>().ToArray()),
        ("epoch",           Enumerable.Range(200, 500).Cast<object>().ToArray()),
        ("learning_rate",   [0.001, 0.01, 0.1]),
        ("momentum",        [0, 0.8, 0.9])
    ),
    training,
    validation,
    (param) => new ClassicalFeedforwardNetwork(InputSize, param.Get<int>("hidden_size"), OutputSize),
    (param) => new EnumerableBackpropagationTrainer<ClassicalFeedforwardNetwork> {
        EarlyStop = true,

        Epochs = param.Get<int>("epoch"),
        LearningRate = param.Get<int>("learning_rate"),
        MomentumFactor = param.Get<double>("momentum"),
    }
);

var networks = generator.GenerateAndTrain().ToArray();
*/