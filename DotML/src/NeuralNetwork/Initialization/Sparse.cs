namespace DotML.Network.Initialization;

public class SparseInitialization
    : IInitializer
{
    /// <summary>
    /// Percent of neurons to set to non-zero quantities
    /// </summary>
    public double DensePercent {get; private set;}
    
    /// <summary>
    /// Percent of neurons to set to zero
    /// </summary>
    public double SparsePercent => 1.0 - DensePercent;

    /// <summary>
    /// Dense initialization method
    /// </summary>
    public IInitializer Initializer {get; private set;}

    public SparseInitialization(double percentDense, IInitializer denseInitializer) {
        this.DensePercent = Math.Clamp(0, 1.0, percentDense);
        this.Initializer = denseInitializer;
    }

    private static Random rng = new Random();

    public double RandomBias(int input_count, int output_count, int parameterCount) {
        if (rng.NextDouble() < DensePercent) {
            return Initializer.RandomBias(input_count, output_count, parameterCount);
        } 
        return 0;
    }

    public double RandomWeight(int input_count, int output_count, int parameterCount) {
        if (rng.NextDouble() < DensePercent) {
            return Initializer.RandomWeight(input_count, output_count, parameterCount);
        } 
        return 0;
    }
}
