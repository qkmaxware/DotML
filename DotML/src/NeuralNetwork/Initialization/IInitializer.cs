namespace DotML.Network.Initialization;

/// <summary>
/// An initializer to initialize the weights and biases for a Neural Network
/// </summary>
/// <typeparam name="TNetwork">Network type</typeparam>
public interface IInitializer<TNetwork> where TNetwork:INeuralNetwork {

    /// <summary>
    /// Initialize network weights
    /// </summary>
    /// <param name="network">network to initialize</param>
    public void InitializeWeights(TNetwork network);

    /// <summary>
    /// Initialize network bias values
    /// </summary>
    /// <param name="network">network to initialize</param>
    public void InitializeBiases(TNetwork network);

}