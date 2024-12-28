namespace DotML.Network;

/// <summary>
/// Binary step activation function
/// </summary>
public class BinaryStep : ActivationFunction {
    public static readonly ActivationFunction Instance = new BinaryStep();

    public BinaryStep() {}

    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public override double Invoke(double x) {
        return x < 0 ? 0.0 : 1.0;
    }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="y">neuron  output</param>
    /// <returns>derivative result</returns>
    public override double InvokeDerivative(double x) {
        return 0;
    }
}