namespace DotML.Network;

/// <summary>
/// Neural network layer behaviours
/// </summary>
public interface ILayer {
    /// <summary>
    /// Number of inputs into the layer
    /// </summary>
    public int InputCount {get;}

    /// <summary>
    /// Number of outputs from the layer
    /// </summary>
    public int OutputCount {get;}

    /// <summary>
    /// The results of the last evaluation, for debugging
    /// </summary>
    public Vec<double> GetLastOutputs();

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public int TrainableParameterCount();
}

public interface ILayerWithNeurons : ILayer {
    /// <summary>
    /// Size of the layer, number of neurons
    /// </summary>
    public int NeuronCount { get; }
    /// <summary>
    /// Get a reference to a specific neuron
    /// </summary>
    /// <param name="index">neuron index</param>
    /// <returns>neuron reference</returns>
    public INeuron GetNeuron(int index);
    /// <summary>
    /// Apply an action to every neuron in the layer
    /// </summary>
    /// <param name="action">action to apply to each neuron</param>
    public void ForeachNeuron(Action<INeuron> action) {
        for (var i = 0; i < this.NeuronCount; i++) {
            var neuron = GetNeuron(i);
            action(neuron);
        }
    }
}