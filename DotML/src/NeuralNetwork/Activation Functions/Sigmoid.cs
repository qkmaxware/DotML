namespace DotML.Network;

/// <summary>
/// Sigmoid activation function
/// </summary>
public class Sigmoid : ActivationFunction {
    public static readonly ActivationFunction Instance = new Sigmoid();

    public Sigmoid() {}

    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public override double Invoke(double x) {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="x">neuron  output</param>
    /// <returns>derivative result</returns>
    public override double InvokeDerivative(double x) {
        // Actual derivative... Invoke(x) * (1.0 - Invoke(x));
        // Derivative using Invoke(x) as input already
        var y = Invoke(x);
        return y * (1.0 - y); 
    }
}