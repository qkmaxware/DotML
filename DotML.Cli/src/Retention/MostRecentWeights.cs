namespace DotML.Cli.Retention;

/// <summary>
/// Retention policy to save only the most recent 'N' weights
/// </summary>
public class MostRecentWeights  : IRetentionPolicy<Safetensors> {
    private string root_dir;
    private string? last_weights;

    public MostRecentWeights(string dir) {
        this.root_dir = dir;
    }

    public void Backup(string name, Safetensors backup) {
        if (!string.IsNullOrEmpty(last_weights) && File.Exists(last_weights)) {
            File.Delete(last_weights);
        }
        var next_weights = Path.Combine(root_dir, GetType().Name + "." + name);
        backup.WriteToFile(next_weights);
        this.last_weights = next_weights;
    }
}