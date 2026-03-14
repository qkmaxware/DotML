namespace DotML.Network.Initialization;

public class RandomInitialization
    : IInitializer
{

    private float min;
    private float max;
    private static Random rng = new Random();

    public RandomInitialization(float min, float max) {
        this.min = Math.Min(min, max);
        this.max = Math.Max(min, max);
    }

    public float RandomWeight(int input_count, int output_count, int parameterCount) {
        var sample = rng.NextDouble();
        var number = (max * sample) + (min * (1d - sample));
        return (float)number;
    }

    public float RandomBias(int input_count, int output_count, int parameterCount) {
        var sample = rng.NextDouble();
        var number = (max * sample) + (min * (1d - sample));
        return (float)number;
    }
}