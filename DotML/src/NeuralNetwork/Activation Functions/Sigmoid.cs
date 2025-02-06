namespace DotML.Network;

/// <summary>
/// <para>Sigmoid activation function</para>
/// <para>Sigmoid(x) = 1 / (1 + e^-x)</para>
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
    /// <param name="x">neuron  output</param>
    /// <returns>derivative result</returns>
    public override double InvokeDerivative(double x) {
        // Actual derivative... Invoke(x) * (1.0 - Invoke(x));
        // Derivative using Invoke(x) as input already
        var y = Invoke(x);
        return y * (1.0 - y); 
    }

    public override void ToHtml(TextWriter writer) {
        writer.Write( 
$@"<math>
    <mtext>f(x) = </mtext>
    <mfrac>
        <mrow>
            <mtext>x</mtext>
        </mrow>
        <mrow>
            <mtext>1</mtext>
            <mo>+</mo>
            <msup><mi>e</mi><mn>-x</mn></msup>
        </mrow>
    </mfrac>
</math>");
    }
}