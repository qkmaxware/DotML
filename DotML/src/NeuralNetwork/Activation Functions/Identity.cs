namespace DotML.Network;

/// <summary>
/// F(x) = x activation function
/// </summary>
public class Identity : ActivationFunction {
    public static readonly ActivationFunction Instance = new Identity();

    public Identity() {}

    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public override double Invoke(double x) {
        return x;
    }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="x">neuron  output</param>
    /// <returns>derivative result</returns>
    public override double InvokeDerivative(double x) {
        return 1;
    }
}