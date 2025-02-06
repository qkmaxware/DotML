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

    public void NotifyTrainingStarted(FeedforwardNetwork network) {
        SendMessage($"Training beginning for network {network.Name}.");
    }
    public void NotifyTrainingStep(FeedforwardNetwork network, int epoch, int epochs, IValidationReport status) {
        SendMessage(
@$"Training update for network {network.Name}. 

> **Epoch {epoch}/{epochs}**
> Tests: {status.TestsPassedCount}/{status.TestCount}
> Loss: {status.MinLoss}-{status.MaxLoss} (avg: {status.AverageLoss})"
);
    }
    public void NotifyTrainingDone(FeedforwardNetwork network, int epochs, IValidationReport final_status) {
        SendMessage(
@$"Training completed for network {network.Name}. 

> **Epoch epochs**
> Tests: {final_status.TestsPassedCount}/{final_status.TestCount}
> Loss: {final_status.MinLoss}-{final_status.MaxLoss} (avg: {final_status.AverageLoss})"
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