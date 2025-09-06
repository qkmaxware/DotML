namespace DotML.Network;

/// <summary>
/// <para>Parametric rectified linear unit activation function</para>
/// <para>PReLU(x) = a * x if x &lt; 0 else x</para>
/// </summary>
public class PReLU : ActivationFunction {
    //          Preferred      , Uncommon
    // Commonly 0.01, 0.05, 0.1, 0.3, 0.5
    public float Alpha {get; init;}

    public PReLU(float alpha) {
        this.Alpha = alpha;
    }

    public override float Invoke(float x) {
        return x < 0 ? Alpha * x : x;
    }

    public override float InvokeDerivative(float x) {
        return x < 0 ? Alpha : 1f;
    }

    public override string ToString() {
        return base.ToString() + $"({Alpha})";
    }

    public override void ToHtml(TextWriter writer) {
        writer.Write( 
$@"<math>
    <mtext>f(x) = </mtext>
    <mrow>
        <mo>{{</mo>
        <mtable>
            <mtr>
                <mtd>{Alpha} x</mtd>
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