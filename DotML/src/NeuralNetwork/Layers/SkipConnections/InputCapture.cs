using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// A layer which captures (caches) it's inputs so that other layers like those use for skip connections can reference the inputs later
/// </summary>
[WorkInProgress]
public class InputCapture : FeedforwardNetworkLayer {

    public BatchedFeatureSet<double>? CapturedInput;

    public InputCapture(Shape3D input_shape) {
        this.InputShape = input_shape;
        this.OutputShape = input_shape;
    }

    /// <summary>
    /// Cache the given feature set
    /// </summary>
    /// <param name="channels">feature set</param>
    /// <returns>the cached feature set</returns>
    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        CapturedInput = new BatchedFeatureSet<double>(channels);
        return channels;
    }

    /// <summary>
    /// Cache the given feature set
    /// </summary>
    /// <param name="features">feature set</param>
    /// <returns>the cached feature set</returns>
    public override BatchedFeatureSet<double> EvaluateSync(BatchedFeatureSet<double> features) {
        CapturedInput = features;
        return features;
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        return new BackpropagationReturns(args.OutputErrors, null);
    }

    public override void SubtractGradients(LayerGradients? gradients) { }

    public override void Initialize(IInitializer initializer) { }

    public override int TrainableParameterCount() { return 0; }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);

    public override T Visit<T>(ILayerVisitor<T> visitor) => visitor.Visit(this);

    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}