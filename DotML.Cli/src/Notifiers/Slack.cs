using System.Text;
using System.Text.Json;
using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Notifiers;

public class SlackFactory : INotifierFactory {
    public bool SupportsEndpoint(string endpoint) => endpoint.StartsWith("https://hooks.slack.com/services/");

    public INotifier Make(string endpoint) => new Slack(endpoint);
}

/// <summary>
/// Notifier to send update messages during training to Discord via a webhook
/// </summary>
public class Slack : INotifier {

    private string webhook_url;

    public Slack(string webhook) {
        this.webhook_url = webhook;
    }

    public void NotifyTrainingStarted(INetworkModule network) {
        SendMessage($"Training beginning for network {network.Name()}.");
    }
    public void NotifyTrainingStep(INetworkModule network, int epoch, int epochs, IValidationReport status) {
        SendMessage(
@$"Training update for network {network.Name()}. 

> **Epoch {epoch}/{epochs}**
> Tests: {status.TestsPassedCount}/{status.SampleCount}
> Loss: {status.Loss.Min}-{status.Loss.Max} (avg: {status.Loss.Average})"
);
    }
    public void NotifyNewBest(INetworkModule network, int epoch, int epochs, IValidationReport status) {
        SendMessage(
@$"New best weights found for network {network.Name()}. 

> **Epoch {epoch}/{epochs}**
> Tests: {status.TestsPassedCount}/{status.SampleCount}
> Loss: {status.Loss.Min}-{status.Loss.Max} (avg: {status.Loss.Average})"
);
    }
    public void NotifyTrainingDone(INetworkModule network, int epochs, IValidationReport final_status) {
        SendMessage(
@$"Training completed for network {network.Name()}. 

> **Epoch epochs**
> Tests: {final_status.TestsPassedCount}/{final_status.SampleCount}
> Loss: {final_status.Loss.Min}-{final_status.Loss.Max} (avg: {final_status.Loss.Average})"
);
    }

    public void NotifyTrainingCancelled(INetworkModule network, int epochs) {
        SendMessage(
@$"Training cancelled by user for network {network.Name()} on epoch {epochs}."
);
    }

    private bool SendMessage(string message) {
        using (HttpClient client = new HttpClient()) {
            var payload = new {
                text = string.Empty,
                blocks = new []{
                    new {
                        type = "section",
                        text = new {
                            type = "mrkdwn",
                            text = message
                        }
                    }
                }
            };

            // Convert the payload object to JSON
            string jsonPayload = JsonSerializer.Serialize(payload);
            var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

            try {
                // Send the POST request
                var response = client.PostAsync(webhook_url, content);
                response.Wait();
                return response.Result.StatusCode == System.Net.HttpStatusCode.OK;
            }
            catch {
                return false;
            }
        }
    }
}