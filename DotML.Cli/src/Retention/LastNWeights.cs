namespace DotML.Cli.Retention;

public class LastNWeights : IRetentionPolicy<Safetensors> {
    private string root_dir;
    private int count;
    private LinkedList<string> last_weights = new LinkedList<string>();

    public LastNWeights(string dir, int count = 5) {
        this.root_dir = dir;
        this.count = count;
    }

    public void Backup(string name, Safetensors backup) {
        // Save weights
        var next_weights = Path.Combine(root_dir,  name);
        backup.WriteToFile(next_weights);
        last_weights.AddLast(next_weights);

        // Delete the oldest until we reach the count
        while (last_weights.Count > count) {
            var last_weight = last_weights.First?.Value ?? string.Empty;
            last_weights.RemoveFirst();

            if (!string.IsNullOrEmpty(last_weight) && File.Exists(last_weight)) {
                File.Delete(last_weight);
            }
        }
    }
}