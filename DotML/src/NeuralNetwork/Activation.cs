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
    /// Invoke the activation function on all values in the given vector
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public virtual Matrix<double> Invoke(Matrix<double> xs) => xs.Map(x => Invoke(x));
    // /// <summary>
    // /// Invoke the activation function on all values in the given vector
    // /// </summary>
    // /// <param name="x">function input</param>
    // /// <returns>function result</returns>
    // public virtual Vec<double> Invoke(Vec<double> xs) {
    //     double[] values = new double[xs.Dimensionality];
    //     for (var i = 0; i < values.Length; i++) {
    //         values[i] = Invoke(xs[i]);
    //     }
    //     return Vec<double>.Wrap(values);
    // }
    // /// <summary>
    // /// Invoke the activation function on all values in the given vector
    // /// </summary>
    // /// <param name="x">function input</param>
    // /// <returns>function result</returns>
    // public virtual Matrix<double> Invoke(Matrix<double> xs) {
    //     double[,] values = new double[xs.Rows, xs.Columns];
    //     for (var j = 0; j < xs.Rows; j++) {
    //         for (var i = 0; i < xs.Columns; i++) {
    //             values[j, i] = Invoke(xs[j, i]);
    //         }
    //     }
    //     return Matrix<double>.Wrap(values);
    // }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="y">neuron output</param>
    /// <returns>derivative result</returns>
    public abstract double InvokeDerivative(double y);

    public virtual Matrix<double> InvokeDerivative(Matrix<double> dA, Matrix<double> Z) => Z.Map(y => InvokeDerivative(y));
    // /// <summary>
    // /// Invoke the derivative of the activation function on all output values in the given vector
    // /// </summary>
    // /// <param name="x">neuron outputs</param>
    // /// <returns>function result</returns>
    // public virtual Vec<double> InvokeDerivative(Vec<double> ys) {
    //     double[] values = new double[ys.Dimensionality];
    //     for (var i = 0; i < values.Length; i++) {
    //         values[i] = InvokeDerivative(ys[i]);
    //     }
    //     return Vec<double>.Wrap(values);
    // }
    // /// <summary>
    // /// Invoke the derivative of the activation function on all output values in the given vector
    // /// </summary>
    // /// <param name="x">neuron outputs</param>
    // /// <returns>function result</returns>
    // public virtual Matrix<double> InvokeDerivative(Matrix<double> ys) {
    //     double[,] values = new double[ys.Rows, ys.Columns];
    //     for (var j = 0; j < ys.Rows; j++) {
    //         for (var i = 0; i < ys.Columns; i++) {
    //             values[j, i] = InvokeDerivative(ys[j, i]);
    //         }
    //     }
    //     return Matrix<double>.Wrap(values);
    // }

    public override string ToString() => GetType().Name;
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

public class LeakyReLU : ActivationFunction {

    public static readonly ActivationFunction Instance = new LeakyReLU();

    public override double Invoke(double x) {
        return x <= 0 ? 0.01 * x : x;
    }

    public override double InvokeDerivative(double y) {
        return y <= 0 ? 0.01 : 1;
    }
}

public class PReLU : ActivationFunction {
    //          Preferred      , Uncommon
    // Commonly 0.01, 0.05, 0.1, 0.3, 0.5
    public double Alpha {get; init;}

    public PReLU(double alpha) {
        this.Alpha = alpha;
    }

    public override double Invoke(double x) {
        return x < 0 ? Alpha * x : x;
    }

    public override double InvokeDerivative(double y) {
        return y < 0 ? Alpha : 1;
    }

    public override string? ToString() {
        return base.ToString() + $"({Alpha})";
    }
}

public class ExponentialLU : ActivationFunction {
    public double Alpha {get; init;}

    public ExponentialLU (double alpha) {
        this.Alpha = alpha;
    }

    public override double Invoke(double x) {
        return x < 0 ? Alpha * (Math.Exp(x) - 1) : x;
    }

    public override double InvokeDerivative(double y) {
        return y < 0 ? y + Alpha : 1;
    }

    public override string? ToString() {
        return base.ToString() + $"({Alpha})";
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
    /// <param name="y">neuron  output</param>
    /// <returns>derivative result</returns>
    public override double InvokeDerivative(double y) {
        // Actual derivative... 1 - Invoke(x)^2
        // Derivative using Invoke(x) as input already
        return 1 - y*y;
    }
}