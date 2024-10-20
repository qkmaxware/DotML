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
    /// Size of the layer, number of neurons
    /// </summary>
    public int NeuronCount { get; }

    /// <summary>
    /// The results of the last evaluation, for debugging
    /// </summary>
    public Vec<double> GetLastOutputs();

    /// <summary>
    /// Get a reference to a specific neuron
    /// </summary>
    /// <param name="index">neuron index</param>
    /// <returns>neuron reference</returns>
    public INeuron GetNeuron(int index);
}