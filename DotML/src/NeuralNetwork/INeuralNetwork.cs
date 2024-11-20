using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Generic interface for all neural network implementations
/// </summary>
public interface INeuralNetwork {

    /// <summary>
    /// The number of input values (size of input vector) allowed by this network.
    /// </summary>
    public int InputCount {get;}

    /// <summary>
    /// The number of output values (size of output vector) created by evaluating this network.
    /// </summary>
    public int OutputCount {get;}

    /// <summary>
    /// Initialize the weights and biases of the neural network
    /// </summary>
    /// <param name="initializer">Initializer to produce values</param>
    public void Initialize(IInitializer initializer);

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public int TrainableParameterCount();

    /// <summary>
    /// Run a prediction against the network synchronously
    /// </summary>
    /// <param name="input">vectorized input</param>
    /// <returns>vectorized network output</returns>
    public Vec<double> PredictSync(Vec<double> input);
}

/// <summary>
/// A network that is composed of discrete layers
/// </summary>
public interface ILayeredNeuralNetwork<TLayer> : INeuralNetwork where TLayer: ILayer {
    /// <summary>
    /// Number of layers in this neural network.
    /// </summary>
    public int LayerCount { get; }
    
    /// <summary>
    /// Get a particular layer by index
    /// </summary>
    /// <param name="index">index to layer</param>
    /// <returns>layer</returns>
    public TLayer GetLayer(int index);

    /// <summary>
    /// Gets the first hidden layer in the network 
    /// </summary>
    /// <returns>layer</returns>
    public TLayer GetFirstLayer();

    /// <summary>
    /// Gets the last layer in the network where outputs are produced
    /// </summary>
    /// <returns>layer</returns>
    public TLayer GetOutputLayer();

    /// <summary>
    /// Perform an action on each layer in the network
    /// </summary>
    /// <param name="action">action to perform</param>
    public void ForeachLayer(Action<TLayer> action) {
        var layers = this.LayerCount;
        for (var layer_index = 0; layer_index < layers; layer_index++) {
            var layer = this.GetLayer(layer_index);
            action(layer);
        }
    }
}