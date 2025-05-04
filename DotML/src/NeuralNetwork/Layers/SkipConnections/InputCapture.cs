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
    /// A unique identifier to this input capture
    /// </summary>
    /// <returns>uid</returns>
    public int UID() {
        return base.GetHashCode();
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
    
    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => visitor.Visit(this, args);

    public override T Visit<T>(ILayerOutputVisitor<T> visitor) => visitor.Visit(this);

    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}