namespace DotML.Network;

/// <summary>
/// <para>Sinusoid activation function</para>
/// <para>Sinusoid(x) = sin(x)</para>
/// </summary>
public class Sinusoid : ActivationFunction {
    public static readonly ActivationFunction Instance = new Sinusoid();

    public Sinusoid() {}

    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public override double Invoke(double x) {
        return Math.Sin(x);
    }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="y">neuron  output</param>
    /// <returns>derivative result</returns>
    public override double InvokeDerivative(double x) {
        return Math.Cos(x);
    }

    public override void ToHtml(TextWriter writer) {
        writer.Write(
$@"<math>
    <mtext>f(x) = </mtext>
    <mrow><mtext>sin(x)</mtext></mrow>
</math>");
    }
}