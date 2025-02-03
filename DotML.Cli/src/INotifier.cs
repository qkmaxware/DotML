using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli;

public interface INotifierFactory {
    public bool SupportsEndpoint(string endpoint);
    public INotifier Make(string endpoint);
}

public interface INotifier {
    public void NotifyTrainingStarted(FeedforwardNetwork network);
    public void NotifyTrainingStep(FeedforwardNetwork network, int epoch, int epochs, IValidationReport status);
    public void NotifyTrainingDone(FeedforwardNetwork network, int epochs, IValidationReport final_status);
}