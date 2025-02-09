namespace DotML.Cli.Retention;

/// <summary>
/// Retention policy to save none of the weights
/// </summary>
public class NoWeights : IRetentionPolicy<Safetensors> {

    public NoWeights() {}

    public void Backup(string name, Safetensors backup) {
        // Do nothing
    }
}