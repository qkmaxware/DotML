namespace DotML.Network.Training;

/// <summary>
/// Neural Network trainer behaviours
/// </summary>
/// <typeparam name="TNetwork">Type of network to train</typeparam>
public interface ITrainer<TNetwork> where TNetwork:INeuralNetwork {
    /// <summary>
    /// Train the network against the given dataset.
    /// </summary> 
    /// <param name="network">network to train</param>
    /// <param name="dataset">dataset to train on</param>
    public void Train(TNetwork network, IEnumerator<TrainingPair> dataset) => Train(network, dataset, dataset);
    
    /// <summary>
    /// Train the network against the given dataset and validate against a separate dataset.
    /// </summary>
    /// <param name="network">network to train</param>
    /// <param name="dataset">dataset to train on</param>
    /// <param name="validation">dataset to validate against</param>
    public void Train(TNetwork network, IEnumerator<TrainingPair> dataset, IEnumerator<TrainingPair> validation);
}

/// <summary>
/// Neural Network trainer behaviours for training that can be iterated over rather than all at once
/// </summary>
/// <typeparam name="TNetwork">Type of network to train<</typeparam>
public interface IEnumerableTrainer<TNetwork> : ITrainer<TNetwork> where TNetwork:INeuralNetwork {
    /// <summary>
    /// Fetch an enumerator which can be used to train the network step by step.
    /// </summary>
    /// <param name="network">network to train</param>
    /// <param name="dataset">dataset to train on</param>
    /// <returns>enumerator for step-by-step training</returns>
    public IEpochEnumerator<TNetwork> EnumerateTraining(TNetwork network, IEnumerator<TrainingPair> dataset) => EnumerateTraining(network, dataset, dataset);
    
    /// <summary>
    /// Fetch an enumerator which can be used to train the network step by step.
    /// </summary>
    /// <param name="network">network to train</param>
    /// <param name="dataset">dataset to train on</param>
    /// <param name="validation">dataset to validate against</param>
    /// <returns>enumerator for step-by-step training</returns>
    public IEpochEnumerator<TNetwork> EnumerateTraining(TNetwork network, IEnumerator<TrainingPair> dataset, IEnumerator<TrainingPair> validation);
}