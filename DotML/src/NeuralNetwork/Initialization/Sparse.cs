namespace DotML.Network.Initialization;

public class SparseInitialization
    : IInitializer
{
    /// <summary>
    /// Percent of neurons to set to non-zero quantities
    /// </summary>
    public float DensePercent {get; private set;}
    
    /// <summary>
    /// Percent of neurons to set to zero
    /// </summary>
    public float SparsePercent => 1.0f - DensePercent;

    /// <summary>
    /// Dense initialization method
    /// </summary>
    public IInitializer Initializer {get; private set;}

    public SparseInitialization(float percentDense, IInitializer denseInitializer) {
        this.DensePercent = Math.Clamp(0, 1.0f, percentDense);
        this.Initializer = denseInitializer;
    }

    private static Random rng = new Random();

    public float RandomBias(int input_count, int output_count, int parameterCount) {
        if (rng.NextDouble() < DensePercent) {
            return Initializer.RandomBias(input_count, output_count, parameterCount);
        } 
        return 0;
    }

    public float RandomWeight(int input_count, int output_count, int parameterCount) {
        if (rng.NextDouble() < DensePercent) {
            return Initializer.RandomWeight(input_count, output_count, parameterCount);
        } 
        return 0;
    }
}
