using System.Runtime.CompilerServices;

namespace DotML.Network.Initialization;

public class HeInitialization
    : IInitializer
{
    private static Random rng = new Random();

    public float RandomBias(int input_count, int output_count, int parameterCount) {
        return 0.01f;
    }

    private float NextGaussian() {
        double u1 = rng.NextDouble();
        double u2 = rng.NextDouble();
        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return (float)z0;
    }

    public float RandomWeight(int input_count, int output_count, int parameterCount) {
        float stddev = MathF.Sqrt(2.0f / (input_count + output_count));
        return NextGaussian() * stddev;
    }
}