namespace DotML.Cli.Retention;

/// <summary>
/// Retention policy to save all weights 
/// </summary>
public class AllWeights : IRetentionPolicy<Safetensors> {

    private string root_dir;

    public AllWeights(string dir) {
        this.root_dir = dir;
    }

    public void Backup(string name, Safetensors backup) {
        backup.WriteToFile(Path.Combine(root_dir,  name));
    }
}