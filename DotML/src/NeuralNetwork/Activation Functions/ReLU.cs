namespace DotML.Network;

/// <summary>
/// Rectified linear unit activation function
/// </summary>
public class ReLU : ActivationFunction {
    public static readonly ActivationFunction Instance = new ReLU();

    public ReLU() {}

    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public override double Invoke(double x) {
        return Math.Max(0,  x);
    }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="y">neuron  output</param>
    /// <returns>derivative result</returns>
    public override double InvokeDerivative(double x) {
        return x <= 0 ? 0 : 1;
    }
}