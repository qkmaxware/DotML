namespace DotML.Network;

/// <summary>
/// <para>Identity activation function</para>
/// <para>Identity(x) = x</para>
/// </summary>
public class Identity : ActivationFunction {
    public static readonly ActivationFunction Instance = new Identity();

    public Identity() {}

    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public override float Invoke(float x) {
        return x;
    }
    /// <summary>
    /// Invoke the derivative of the activation function with the given output from the neuron
    /// </summary>
    /// <param name="x">neuron  output</param>
    /// <returns>derivative result</returns>
    public override float InvokeDerivative(float x) {
        return 1;
    }

    public override void ToHtml(TextWriter writer) {
        writer.Write(
$@"<math>
    <mtext>f(x) = x</mtext>
</math>");
    }
}