namespace DotML.Network;

/// <summary>
/// <para>Exponential rectified linear unit activation function</para>
/// <para>ELU(x) = a * (e^x - 1) if x &lt; 0 else x</para>
/// </summary>
public class ExponentialLU : ActivationFunction {
    public double Alpha {get; init;}

    public ExponentialLU (double alpha) {
        this.Alpha = alpha;
    }

    public override double Invoke(double x) {
        return x < 0 ? Alpha * (Math.Exp(x) - 1) : x;
    }

    public override double InvokeDerivative(double x) {
        return x < 0 ? Alpha*Math.Exp(x) : 1;
    }

    public override string ToHtml() {
        return 
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
</math>";
    }

    public override string ToString() {
        return base.ToString() + $"({Alpha})";
    }
}