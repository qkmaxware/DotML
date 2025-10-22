namespace DotML.Network;

/// <summary>
/// <para>Gaussian Error Linear Unit activation function</para>
/// <para>GELU(x) = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))</para>
/// </summary>
public class GELU : ActivationFunction {
    public static readonly ActivationFunction Instance = new TeLU();

    public GELU() {}

    public override float Invoke(float x)
    {
        const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        return 0.5f * x * (1.0f + MathF.Tanh(inner));
    }

    public override float InvokeDerivative(float x)
    {
        // Approximate derivative of GELU(x)
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float x3 = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x3);
        float tanh_val = MathF.Tanh(tanh_arg);

        float left = 0.5f * tanh_val;
        float sech2 = 1 - tanh_val * tanh_val;
        float dx_inner = sqrt_2_over_pi * (1 + 3 * 0.044715f * x * x);
        float right = 0.5f * x * sech2 * dx_inner;
        return left + right + 0.5f;
    }


    public override void ToHtml(TextWriter writer) {
        writer.Write(
$@"<math>
    <mtext>f(x) = </mtext>
    <mn>0.5</mn>
    <mo>*<mo>
    <mrow>
        <mo>(<mo>
        <mn>1</mn>
        <mo>+</mo>
        <mrow>
            <mi>tanh</mi>
            <mo>(</mo>
            <msqrt>
                <mfrac>
                    <mrow>
                        <mn>2</mn>
                    </mrow>
                    <mrow>
                        <mi>pi</mi>
                    </mrow>
                </mfrac>
            </msqrt>
            <mo>*</mo>
            <mrow>
                <mo>(</mo>
                <mi>x</mi>
                <mo>+</mo>
                <mn>0.044715</mn>
                <mo>*</mo>
                <mrow>
                    <mi>x</mi>
                    <msup><mn>3</mn></msup>
                </mrow>
                <mo>)</mo>
            </mrow>
            <mo>)</mo>
        </mrow>
        <mo>)<mo>
    </mrow>
</math>");
    }
}