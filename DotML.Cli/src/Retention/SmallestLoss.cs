using DotML.Network.Training;

namespace DotML.Cli.Retention;

/// <summary>
/// Retention policy to save only the weights with the smallest loss value
/// </summary>
public class SmallestLoss  : IRetentionPolicy<Safetensors> {
    private string root_dir;
    private IValidationReport report;
    private string? last_weights; double? last_weights_loss;

    public SmallestLoss(string dir, IValidationReport report) {
        this.root_dir = dir;
        this.report = report;
    }

    public void Backup(string name, Safetensors backup) {
        if (last_weights_loss.HasValue && report.AverageLoss >= last_weights_loss.Value)
            return; // Not better than the last option

        if (!string.IsNullOrEmpty(last_weights) && File.Exists(last_weights)) {
            File.Delete(last_weights);
        }
        var next_weights = Path.Combine(root_dir, GetType().Name + "." + name);
        backup.WriteToFile(next_weights);
        this.last_weights = next_weights;
        this.last_weights_loss = report.AverageLoss;
    }
}