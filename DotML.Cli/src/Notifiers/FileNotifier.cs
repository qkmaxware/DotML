using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Notifiers;

public class FileFactory : INotifierFactory {
    private static Regex file_pattern = new Regex(@"^file://");

    public bool SupportsEndpoint(string endpoint) => file_pattern.IsMatch(endpoint);

    public INotifier Make(string endpoint) => new FileNotifier(file_pattern.Replace(endpoint, string.Empty));
}

/// <summary>
/// Notifier to send update messages during training to Discord via a webhook
/// </summary>
public class FileNotifier : INotifier {

    private string file_path;

    public FileNotifier(string path) {
        this.file_path = path;
    }

    public void NotifyTrainingStarted(INetworkModule network) {
        SendMessage($"Training beginning for network {network.Name()}.");
    }
    public void NotifyTrainingStep(INetworkModule network, int epoch, int epochs, IValidationReport status) {
        SendMessage(
@$"Training update for network {network.Name()}. 

> **Epoch {epoch}/{epochs}**
> Tests: {status.TestsPassedCount}/{status.SampleCount}
> Loss: {status.MinLoss}-{status.MaxLoss} (avg: {status.AvgLoss})"
);
    }
    public void NotifyNewBest(INetworkModule network, int epoch, int epochs, IValidationReport status) {
        SendMessage(
@$"New best weights found for network {network.Name()}. 

> **Epoch {epoch}/{epochs}**
> Tests: {status.TestsPassedCount}/{status.SampleCount}
> Loss: {status.MinLoss}-{status.MaxLoss} (avg: {status.AvgLoss})"
);
    }
    public void NotifyTrainingDone(INetworkModule network, int epochs, IValidationReport final_status) {
        SendMessage(
@$"Training completed for network {network.Name()}. 

> **Epoch epochs**
> Tests: {final_status.TestsPassedCount}/{final_status.SampleCount}
> Loss: {final_status.MinLoss}-{final_status.MaxLoss} (avg: {final_status.AvgLoss})"
);
    }

    public void NotifyTrainingCancelled(INetworkModule network, int epochs) {
        SendMessage(
@$"Training cancelled by user for network {network.Name()} on epoch {epochs}."
);
    }

    private bool SendMessage(string message) {
        using (var writer = new StreamWriter(File.Open(this.file_path, FileMode.Append))) {
            writer.Write(message);
        }
        return true;
    }
}