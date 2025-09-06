namespace DotML.Network;

/// <summary>
/// Neuron behaviours
/// </summary>
public interface INeuron {
    /// <summary>
    /// Neuron bias value
    /// </summary>
    /// <value>bias</value>
    public float Bias {get; set;}
    /// <summary>
    /// Neuron input weights
    /// </summary>
    /// <value>vector of weights, one per input</value>
    public Span<float> Weights {get;}
    /// <summary>
    /// Activation function for outputs
    /// </summary>
    /// <value>activation function or null</value>
    public ActivationFunction? ActivationFunction {get;}
}

public interface IIndividuallyActivatedNeuron : INeuron {
    /// <summary>
    /// Activation function for outputs
    /// </summary>
    /// <value>activation function or null</value>
    public new ActivationFunction? ActivationFunction {get; set;}
}