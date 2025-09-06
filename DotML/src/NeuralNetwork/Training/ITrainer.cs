using System.Numerics;

namespace DotML.Network.Training;

/// <summary>
/// Neural Network trainer behaviours
/// </summary>
/// <typeparam name="TNetwork">Type of network to train</typeparam>
public interface ITrainer<TNetwork, TVector> where TNetwork:INeuralNetwork where TVector:INumber<TVector> {
    /// <summary>
    /// Train the network against the given dataset.
    /// </summary> 
    /// <param name="network">network to train</param>
    /// <param name="dataset">dataset to train on</param>
    public void Train(TNetwork network, IEnumerator<TrainingPair<TVector>> dataset) => Train(network, dataset, dataset);
    
    /// <summary>
    /// Train the network against the given dataset and validate against a separate dataset.
    /// </summary>
    /// <param name="network">network to train</param>
    /// <param name="dataset">dataset to train on</param>
    /// <param name="validation">dataset to validate against</param>
    public void Train(TNetwork network, IEnumerator<TrainingPair<TVector>> dataset, IEnumerator<TrainingPair<TVector>> validation);
}

/// <summary>
/// Neural Network trainer behaviours for training that can be iterated over rather than all at once
/// </summary>
/// <typeparam name="TNetwork">Type of network to train<</typeparam>
public interface IEnumerableTrainer<TNetwork, TVector> : ITrainer<TNetwork, TVector> where TNetwork:INeuralNetwork where TVector:INumber<TVector> {
    /// <summary>
    /// Gets or sets a place to report testing validation results to
    /// </summary>
    public IValidationReport? ValidationReport {get; set;}

    /// <summary>
    /// Fetch an enumerator which can be used to train the network step by step.
    /// </summary>
    /// <param name="network">network to train</param>
    /// <param name="dataset">dataset to train on</param>
    /// <returns>enumerator for step-by-step training</returns>
    public IEpochEnumerator<TNetwork> EnumerateTraining(TNetwork network, IEnumerator<TrainingPair<TVector>> dataset) => EnumerateTraining(network, dataset, dataset);
    
    /// <summary>
    /// Fetch an enumerator which can be used to train the network step by step.
    /// </summary>
    /// <param name="network">network to train</param>
    /// <param name="dataset">dataset to train on</param>
    /// <param name="validation">dataset to validate against</param>
    /// <returns>enumerator for step-by-step training</returns>
    public IEpochEnumerator<TNetwork> EnumerateTraining(TNetwork network, IEnumerator<TrainingPair<TVector>> dataset, IEnumerator<TrainingPair<TVector>> validation);
}