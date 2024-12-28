namespace DotML.Network;

/// <summary>
/// Hyperbolic tangent activation function
/// </summary>
public class HyperbolicTangent : ActivationFunction {
    public static readonly ActivationFunction Instance = new HyperbolicTangent();

    public HyperbolicTangent() {}

    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public override double Invoke(double x) {
        var result = Math.Tanh(x);
        return result;
        /*if (double.IsPositiveInfinity(x))
            return 1;
        if (double.IsNegativeInfinity(x))
            return -1;
        var pex = Math.Exp(x);
        var nex = Math.Exp(-x); 
        var result = (pex - nex) / (pex + nex + double.Epsilon);
        return result;*/
    }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="x">neuron output</param>
    /// <returns>derivative result</returns>
    public override double InvokeDerivative(double x) {
        // Actual derivative... 1 - Invoke(x)^2
        // Derivative using Invoke(x) as input already
        var y = Invoke(x);
        return 1 - y*y;
    }
}