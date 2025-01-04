namespace DotML.Network;

/// <summary>
/// Exponential rectified linear unit activation function
/// </summary>
/// <remarks>
/// ELU(x) = a * (e^x - 1) if x < 0 else x
/// </remarks>
public class ExponentialLU : ActivationFunction {
    public double Alpha {get; init;}

    public ExponentialLU (double alpha) {
        this.Alpha = alpha;
    }

    public override double Invoke(double x) {
        return x < 0 ? Alpha * (Math.Exp(x) - 1) : x;
    }

    public override double InvokeDerivative(double x) {
        return x < 0 ? Alpha*Math.Exp(x) : 1;
    }

    public override string ToString() {
        return base.ToString() + $"({Alpha})";
    }
}