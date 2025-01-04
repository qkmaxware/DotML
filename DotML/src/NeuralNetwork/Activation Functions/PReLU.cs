namespace DotML.Network;

/// <summary>
/// Parametric rectified linear unit activation function
/// </summary>
/// <remarks>
/// PReLU(x) = a * x if x < 0 else x
/// </remarks>
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

    public override double InvokeDerivative(double x) {
        return x < 0 ? Alpha : 1;
    }

    public override string ToString() {
        return base.ToString() + $"({Alpha})";
    }
}