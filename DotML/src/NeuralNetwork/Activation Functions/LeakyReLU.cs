namespace DotML.Network;

/// <summary>
/// Leaky rectified linear unit activation function
/// </summary>
public class LeakyReLU : ActivationFunction {

    public static readonly ActivationFunction Instance = new LeakyReLU();

    public override double Invoke(double x) {
        return x <= 0 ? 0.01 * x : x;
    }

    public override double InvokeDerivative(double x) {
        return x <= 0 ? 0.01 : 1;
    }
}