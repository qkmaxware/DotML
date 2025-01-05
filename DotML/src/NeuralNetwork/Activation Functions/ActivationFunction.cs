namespace DotML.Network;

/// <summary>
/// Neuron activation function
/// <see href="https://en.wikipedia.org/wiki/Activation_function"/>
/// </summary>
public abstract class ActivationFunction : IHtmlable {
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
    public virtual string ToHtml() {
        return
$@"<math>
    <mtext>f(x) = </mtext>
    <mtext>{ToString()}(x)</mtext>
</math>";
    }

    public override string ToString() => GetType().Name;
}
