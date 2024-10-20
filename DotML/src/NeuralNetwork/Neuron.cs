using System.Text.Json.Serialization;

namespace DotML.Network;

/// <summary>
/// Simple struct based neuron
/// </summary>
public struct Neuron : INeuron {
    /// <summary>
    /// Neuron bias value
    /// </summary>
    /// <value>bias</value>
    public double Bias {get; set;}
    /// <summary>
    /// Neuron input weights
    /// </summary>
    /// <value>vector of weights, one per input</value>
    public double[] Weights {get; set;}
    [JsonIgnore] Span<double> INeuron.Weights => Weights == null ? Span<double>.Empty : Weights;
    /// <summary>
    /// Activation function for outputs
    /// </summary>
    /// <value>activation function or null</value>
    [JsonIgnore] public ActivationFunction? ActivationFunction {get; set;}
    /// <summary>
    /// Name of the activation function
    /// </summary>
    /// <value>the fully qualified name of the activation function</value>
    public string? ActivationFunctionClass {
        get {
            return ActivationFunction?.GetType()?.AssemblyQualifiedName;
        } set {
            if (value is null) {
                ActivationFunction = Identity.Instance;
                return;
            }
            Type? type = Type.GetType(value);
            if (type is null) {
                ActivationFunction = Identity.Instance;
                return;
            }
            if (!type.IsAssignableTo(typeof(ActivationFunction))) {
                ActivationFunction = Identity.Instance;
                return;
            }
            try {
                var instance = Activator.CreateInstance(type);
                if (instance is null)   
                    this.ActivationFunction = Identity.Instance;
                else
                    this.ActivationFunction = (ActivationFunction)instance;
            } catch {
                this.ActivationFunction = Identity.Instance;
            }
        }
    }

    /// <summary>
    /// Get the weight of a given input.
    /// </summary>
    /// <param name="i">input index</param>
    /// <returns>weight if given, or 0 if no matching weight exists</returns>
    public readonly double Weight(int i) {
        if (i < 0 || i >= Weights.Length)
            return 0.0;
        return Weights[i];
    }

    /// <summary>
    /// Evaluate this neuron's activation based on the given input vector
    /// </summary>
    /// <param name="inputs">input vector</param>
    /// <returns>neuron output</returns>
    public readonly double Evaluate(Vec<double> inputs) {
        var activation = Bias;
        for (var i = 0; i < inputs.Dimensionality; i++) {
            activation += inputs[i] * Weight(i);
        }
        if (ActivationFunction is not null) {
            activation = ActivationFunction.Invoke(activation);
        }
        return activation;
    }
}