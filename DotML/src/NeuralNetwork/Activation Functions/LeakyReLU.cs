namespace DotML.Network;

/// <summary>
/// <para>Leaky rectified linear unit activation function</para>
/// <para>LeakyReLU(x) = 0.01 * x if x &lt; 0 else x</para>
/// </summary>
public class LeakyReLU : ActivationFunction {

    public static readonly ActivationFunction Instance = new LeakyReLU();

    public override double Invoke(double x) {
        return x <= 0 ? 0.01 * x : x;
    }

    public override double InvokeDerivative(double x) {
        return x <= 0 ? 0.01 : 1;
    }

    public override void ToHtml(TextWriter writer) {
        writer.Write( 
$@"<math>
    <mtext>f(x) = </mtext>
    <mrow>
        <mo>{{</mo>
        <mtable>
            <mtr>
                <mtd>0.01 x</mtd>
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