using System.Runtime.CompilerServices;

namespace DotML.Network.Initialization;

public class LeCunInitialization
    : IInitializer
{
    private static Random rng = new Random();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected static float NormalRandom(double mean, double stddev) {
        double u1 = rng.NextDouble();
        double u2 = rng.NextDouble();
        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return (float)(mean + z0 * stddev);
    }

    public float RandomBias(int input_count, int output_count, int parameterCount) {
        return NormalRandom(0, 1.0 / Math.Sqrt(input_count));
    }

    public float RandomWeight(int input_count, int output_count, int parameterCount) {
        return NormalRandom(0, 1.0 / Math.Sqrt(input_count));
    }
}