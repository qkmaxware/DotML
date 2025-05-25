namespace DotML.Network;

/// <summary>
/// <para>Exponential rectified linear unit activation function</para>
/// <para>ELU(x) = a * (e^x - 1) if x &lt; 0 else x</para>
/// </summary>
public class ExponentialLU : ActivationFunction {
    public float Alpha {get; init;}

    public ExponentialLU (float alpha) {
        this.Alpha = alpha;
    }

    public override float Invoke(float x) {
        return x < 0 ? Alpha * (MathF.Exp(x) - 1) : x;
    }

    public override float InvokeDerivative(float x) {
        return x < 0 ? Alpha*MathF.Exp(x) : 1;
    }

    public override void ToHtml(TextWriter writer) {
        writer.Write( 
$@"<math>
    <mtext>f(x) = </mtext>
    <mrow>
        <mo>{{</mo>
        <mtable>
            <mtr>
                <mtd>{Alpha} (<msup><mi>e</mi><mn>x</mn></msup> - 1)</mtd>
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

    public override string ToString() {
        return base.ToString() + $"({Alpha})";
    }
}