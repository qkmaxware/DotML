using System.Runtime.CompilerServices;

namespace DotML.Network.Initialization;

public class HeInitialization
    : IInitializer
{
    private static Random rng = new Random();

    public double RandomBias(int input_count, int output_count, int parameterCount) {
        return 0.01;
    }

    private double NextGaussian() {
        double u1 = rng.NextDouble();
        double u2 = rng.NextDouble();
        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return z0;
    }

    public double RandomWeight(int input_count, int output_count, int parameterCount) {
        double stddev = Math.Sqrt(2.0 / (input_count + output_count));
        return NextGaussian() * stddev;
    }
}