namespace DotML.Network;

/// <summary>
/// <para>Binary step activation function</para>
/// <para>Step(x) = 1 if x > 0 else 0</para>
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

    public override void ToHtml(TextWriter writer) {
        writer.Write( 
$@"<math>
    <mtext>f(x) = </mtext>
    <mrow>
        <mo>{{</mo>
        <mtable>
            <mtr>
                <mtd>0</mtd>
                <mtd>if x &lt; 0</mtd>
            </mtr>
            <mtr>
                <mtd>1</mtd>
                <mtd>if x &gt; 0</mtd>
            </mtr>
        </mtable>
    </mrow>
</math>");
    }
}