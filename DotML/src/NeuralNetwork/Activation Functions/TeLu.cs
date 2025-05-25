namespace DotML.Network;

/// <summary>
/// <para>Hyperbolic tangent linear unit activation function</para>
/// <para>TeLU(x) = x * tanh(e^x)</para>
/// </summary>
public class TeLU : ActivationFunction {
    public static readonly ActivationFunction Instance = new TeLU();

    public TeLU() {}

    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public override float Invoke(float x) {
        return x * MathF.Tanh(MathF.Exp(x));
    }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="y">neuron  output</param>
    /// <returns>derivative result</returns>
    public override float InvokeDerivative(float x) {
        // https://www.wolframalpha.com/input?i=derivative+of+x+*+tanh%28exp%28x%29%29
        var ex = MathF.Exp(x);
        var sec = 1.0f / MathF.Cosh(ex); // sech(ex)
        return MathF.Tanh(ex) + ex * x * sec * sec;
    }

    public override void ToHtml(TextWriter writer) {
        writer.Write(
$@"<math>
    <mtext>f(x) = </mtext>
    <mrow><mtext>x tanh(</mtext><msup><mi>e</mi><mn>x</mn></msup><mtext>)</mtext></mrow>
</math>");
    }
}