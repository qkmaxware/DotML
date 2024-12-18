namespace DotML.Network.Initialization;

public interface IInitializer {
    public double RandomWeight(int input_count, int output_count, int parameterCount);
    public double RandomBias(int input_count, int output_count, int parameterCount);
}