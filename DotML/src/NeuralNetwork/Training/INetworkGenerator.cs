using System.Numerics;

namespace DotML.Network.Training;

/// <summary>
/// Object that generates networks
/// </summary>
public interface INetworkGenerator  {
    /// <summary>
    /// Generate a sequence of networks
    /// </summary>
    /// <returns>list of networks</returns>
    public IEnumerable<INetworkModule> Generate();
}

/// <summary>
/// Behaviours for an object that both generates networks and trains them
/// </summary>
public interface ITrainedNetworkGenerator : INetworkGenerator {
    /// <summary>
    /// Generate a sequence of fully trained networks
    /// </summary>
    /// <returns>list of networks all of which have undergone the training process</returns>
    IEnumerable<INetworkModule> GenerateAndTrain();
}

public delegate INetworkModule ParameterizedNetworkFactory(ParameterSet @params);

/// <summary>
/// Class to generate a sequence of networks from a parameter matrix
/// </summary>
public class ParameterizedNetworkGenerator : INetworkGenerator
{
    private ParameterMatrix matrix;
    private ParameterizedNetworkFactory factory;

    public ParameterizedNetworkGenerator(
        ParameterMatrix @params,
        ParameterizedNetworkFactory factory
    )
    {
        this.matrix = @params;
        this.factory = factory;
    }

    /// <summary>
    /// Generate a sequence of networks
    /// </summary>
    /// <returns>list of networks</returns>
    public IEnumerable<INetworkModule> Generate()
    {
        foreach (var set in matrix)
        {
            yield return factory(set);
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