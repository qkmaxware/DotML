namespace DotML.Network;

/// <summary>
/// Neuron activation function
/// <see href="https://en.wikipedia.org/wiki/Activation_function"/>
/// </summary>
public abstract class ActivationFunction : DelegateObject<double, double>, IHtmlable {
    /// <summary>
    /// Invoke the activation function on all values in the given vector
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public virtual Matrix<double> Invoke(Matrix<double> xs) => xs.Transform(x => Invoke(x));

    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="y">neuron output</param>
    /// <returns>derivative result</returns>
    public abstract double InvokeDerivative(double y);
    
    /// <summary>
    /// Invoke the derivative of the activation function on all values in the given vector
    /// </summary>
    /// <param name="Z">function input</param>
    /// <returns>function result</returns>
    public virtual Matrix<double> InvokeDerivative(Matrix<double> xs) => xs.Transform(x => InvokeDerivative(x));
    
    /// <summary>
    /// Activation function as HTML MathML
    /// </summary>
    /// <returns>HTML string</returns>
    public virtual void ToHtml(TextWriter writer) {
        writer.Write(
$@"<math>
    <mtext>f(x) = </mtext>
    <mtext>{ToString()}(x)</mtext>
</math>");
    }

    public override string ToString() => GetType().Name;
}

/// <summary>
/// An easy access list of some activation functions in one location
/// </summary>
public static class ActivationFunctions {
    /// <summary>
    /// Enumerate over all activation functions
    /// </summary>
    /// <returns>enumerable of activation functions</returns>
    public static IEnumerable<ActivationFunction> EnumerateAll() {
        return typeof(ActivationFunctions)
            .GetProperties(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public)
            .Where(prop => prop.CanRead && prop.PropertyType.IsAssignableTo(typeof(ActivationFunction)))
            .Select(prop => prop.GetValue(null))
            .OfType<ActivationFunction>();
    }

    /// <summary>
    /// f(x) = x
    /// </summary>
    public static ActivationFunction Identity => Network.Identity.Instance;
    /// <summary>
    /// f(x) =  1 if x > 0 else 0
    /// </summary>
    public static ActivationFunction BinaryStep => Network.BinaryStep.Instance;
    /// <summary>
    /// f(x) = 0.00 * (e^x - 1) if x &lt; 0 else x
    /// </summary>
    public static ActivationFunction ELU_00 {get; private set;} = new Network.ExponentialLU(0.00);
    /// <summary>
    /// f(x) = 0.10 * (e^x - 1) if x &lt; 0 else x
    /// </summary>
    public static ActivationFunction ELU_01 {get; private set;} = new Network.ExponentialLU(0.10);
    /// <summary>
    /// f(x) = 0.20 * (e^x - 1) if x &lt; 0 else x
    /// </summary>
    public static ActivationFunction ELU_02 {get; private set;} = new Network.ExponentialLU(0.20);
    /// <summary>
    /// f(x) = 1.00 * (e^x - 1) if x &lt; 0 else x
    /// </summary>
    public static ActivationFunction ELU_10 {get; private set;} = new Network.ExponentialLU(1.00);
    /// <summary>
    /// f(x) = tanh(x)
    /// </summary>
    public static ActivationFunction Tanh => Network.HyperbolicTangent.Instance;
    /// <summary>
    /// f(x) = 0.01 * x if x &lt; 0 else x
    /// </summary>
    public static ActivationFunction LeakyReLU => Network.LeakyReLU.Instance;
    /// <summary>
    /// PReLU(x) = 0.01 * x if x &lt; 0 else x
    /// </summary>
    public static ActivationFunction PReLU_01 {get; private set;} = new Network.PReLU(0.01);
    /// <summary>
    /// PReLU(x) = 0.05 * x if x &lt; 0 else x
    /// </summary>
    public static ActivationFunction PReLU_05 {get; private set;} = new Network.PReLU(0.05);
    /// <summary>
    /// PReLU(x) = 0.10 * x if x &lt; 0 else x
    /// </summary>
    public static ActivationFunction PReLU_10 {get; private set;} = new Network.PReLU(0.10);
    /// <summary>
    /// PReLU(x) = 0.30 * x if x &lt; 0 else x
    /// </summary>
    public static ActivationFunction PReLU_30 {get; private set;} = new Network.PReLU(0.30);
    /// <summary>
    /// PReLU(x) = 0.50 * x if x &lt; 0 else x
    /// </summary>
    public static ActivationFunction PReLU_50 {get; private set;} = new Network.PReLU(0.50);
    /// <summary>
    /// f(x) = max(0, x)
    /// </summary>
    public static ActivationFunction ReLU => Network.ReLU.Instance;
    /// <summary>
    /// f(x) = 1 / (1 + e^-x)
    /// </summary>
    public static ActivationFunction Sigmoid => Network.Sigmoid.Instance;
    /// <summary>
    /// f(x) = sin(x)
    /// </summary>
    public static ActivationFunction Sinusoid => Network.Sinusoid.Instance;
    /// <summary>
    /// f(x) = ln(x + e^x)
    /// </summary>
    public static ActivationFunction Softplus => Network.Softplus.Instance;
    /// <summary>
    /// f(x) = x * tanh(e^x)
    /// </summary>
    public static ActivationFunction TeLU => Network.TeLU.Instance;

}