namespace DotML.Network;

/// <summary>
/// Neuron activation function
/// </summary>
public abstract class ActivationFunction {
    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public abstract double Invoke(double x);    
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="y">neuron  output</param>
    /// <returns>derivative result</returns>
    public abstract double InvokeDerivative(double y);
}

// List of activation functions from Wikipedia https://en.wikipedia.org/wiki/Activation_function

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
    /// <param name="y">neuron  output</param>
    /// <returns>derivative result</returns>
    public override double InvokeDerivative(double y) {
        return 1;
    }
}

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
    public override double InvokeDerivative(double y) {
        return y <= 0 ? 0 : 1;
    }
}

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
    /// <param name="y">neuron  output</param>
    /// <returns>derivative result</returns>
    public override double InvokeDerivative(double y) {
        // Actual derivative... Invoke(x) * (1.0 - Invoke(x));
        // Derivative using Invoke(x) as input already
        return y * (1.0 - y); 
    }
}

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
        var pex = Math.Exp(x);
        var nex = Math.Exp(-x); 
        return (pex - nex) / (pex + nex + double.Epsilon);
    }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="y">neuron  output</param>
    /// <returns>derivative result</returns>
    public override double InvokeDerivative(double y) {
        // Actual derivative... 1 - Invoke(x)^2
        // Derivative using Invoke(x) as input already
        return 1 - y*y;
    }
}