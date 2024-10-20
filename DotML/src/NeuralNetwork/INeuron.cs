namespace DotML.Network;

/// <summary>
/// Neuron behaviours
/// </summary>
public interface INeuron {
    /// <summary>
    /// Neuron bias value
    /// </summary>
    /// <value>bias</value>
    public double Bias {get; set;}
    /// <summary>
    /// Neuron input weights
    /// </summary>
    /// <value>vector of weights, one per input</value>
    public Span<double> Weights {get;}
    /// <summary>
    /// Activation function for outputs
    /// </summary>
    /// <value>activation function or null</value>
    public ActivationFunction? ActivationFunction {get; set;}
}