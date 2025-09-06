namespace DotML.Network;

/// <summary>
/// <para>Rectified linear unit activation function</para>
/// <para>ReLU(x) = max(0, x)</para>>
/// </summary>
public class ReLU : ActivationFunction {
    public static readonly ActivationFunction Instance = new ReLU();

    public ReLU() {}

    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public override float Invoke(float x) {
        return MathF.Max(0,  x);
    }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="y">neuron  output</param>
    /// <returns>derivative result</returns>
    public override float InvokeDerivative(float x) {
        return x <= 0 ? 0f : 1f;
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
                <mtd>x</mtd>
                <mtd>if x &gt; 0</mtd>
            </mtr>
        </mtable>
    </mrow>
</math>");
    }
}