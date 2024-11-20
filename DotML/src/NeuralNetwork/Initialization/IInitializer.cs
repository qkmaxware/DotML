namespace DotML.Network.Initialization;

public interface IInitializer {
    public double RandomWeight(int parameterCount);
    public double RandomBias(int parameterCount);
}