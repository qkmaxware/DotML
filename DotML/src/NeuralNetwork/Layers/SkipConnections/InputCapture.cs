using DotML.Network.Initialization;

namespace DotML.Network;

// Desired usage
/*
    new Network (
        ...,
        InputCapture.CaptureInputs(out var capture1),
        ...
        new AdditionSkipConnection(capture1),
        ...
    );
*/

[WorkInProgress]
public class InputCapture : FeedforwardNetworkLayer {

    public BatchedFeatureSet<double>? CapturedInput;

    public static InputCapture CaptureInputs(out InputCapture capture) {
        capture = new InputCapture();
        return capture;
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        CapturedInput = new BatchedFeatureSet<double>(channels);
        return channels;
    }

    public override BatchedFeatureSet<double> EvaluateSync(BatchedFeatureSet<double> features) {
        CapturedInput = features;
        return features;
    }

    public override void Initialize(IInitializer initializer) {
        throw new NotImplementedException();
    }

    public override int TrainableParameterCount() {
        throw new NotImplementedException();
    }

    public override void Visit(ILayerVisitor visitor) {
        throw new NotImplementedException();
    }

    public override T Visit<T>(ILayerVisitor<T> visitor) {
        throw new NotImplementedException();
    }

    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) {
        throw new NotImplementedException();
    }
}