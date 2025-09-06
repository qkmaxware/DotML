namespace DotML.Network;

/// <summary>
/// <para>Softplus activation function</para>
/// <para>Softplus(x) = ln(x + e^x)</para>
/// </summary>
public class Softplus : ActivationFunction {
    public static readonly ActivationFunction Instance = new Softplus();

    public Softplus() {}

    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public override float Invoke(float x) {
        return MathF.Log(1 + MathF.Exp(x));
    }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="y">neuron  output</param>
    /// <returns>derivative result</returns>
    public override float InvokeDerivative(float x) {
        return 1.0f / (1.0f + MathF.Exp(-x));
    }

    public override void ToHtml(TextWriter writer) {
        writer.Write(
$@"<math>
    <mtext>f(x) = </mtext>
    <mrow><mtext>ln(</mtext>1 + <msup><mi>e</mi><mn>x</mn></msup><mtext>)</mtext></mrow>
</math>");
    }
}