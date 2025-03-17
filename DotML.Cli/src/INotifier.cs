using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli;

public enum NotificationEvent : byte {
    none = 0b0000_0000,

    // Basic events
    started = 0b0000_0001,  // Fired when training starts
    step = 0b0000_0010,     // Fired when training finished an epoch
    done = 0b0000_0100,     // Fired when training is done
    cancelled = 0b0000_1000,// Fired when training is cancelled by the user

    // Specific events
    best = 0b0001_0000,     // Fired when a new "best" network is discovered

    // Misc
    all = 0b1111_1111
}

public interface INotifierFactory {
    public bool SupportsEndpoint(string endpoint);
    public INotifier Make(string endpoint);
}

public interface INotifier {
    public void NotifyTrainingStarted(FeedforwardNetwork network);
    public void NotifyTrainingStep(FeedforwardNetwork network, int epoch, int epochs, IValidationReport status);
    public void NotifyNewBest(FeedforwardNetwork network, int epoch, int epochs, IValidationReport status);
    public void NotifyTrainingDone(FeedforwardNetwork network, int epochs, IValidationReport final_status);
    public void NotifyTrainingCancelled(FeedforwardNetwork network, int epochs);
}