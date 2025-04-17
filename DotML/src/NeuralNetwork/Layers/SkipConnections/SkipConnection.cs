using System.Text.RegularExpressions;
using DotML.Network.Initialization;

namespace DotML.Network;

[WorkInProgress]
public abstract class SkipConnection : FeedforwardNetworkLayer {

    /// <summary>
    /// The capture source where the skip connection draws it's features from
    /// </summary>
    public InputCapture CaptureSource {get; private set;}

    public SkipConnection(Shape3D input_shape, Shape3D output_shape, InputCapture captureSource) {
        this.CaptureSource = captureSource;
        this.InputShape = input_shape;
        this.OutputShape = output_shape;
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        return Combine(CaptureSource.CapturedInput, new BatchedFeatureSet<double>(channels))[0];
    }

    public override BatchedFeatureSet<double> EvaluateSync(BatchedFeatureSet<double> features) {
        return Combine(CaptureSource.CapturedInput, features);
    }

    /// <summary>
    /// Combine the features from the capture source with the features passed into this connection as input
    /// </summary>
    /// <param name="skipConnection">capture source features</param>
    /// <param name="input">input features</param>
    /// <returns>combined features</returns>
    public abstract BatchedFeatureSet<double> Combine(BatchedFeatureSet<double>? skipConnection, BatchedFeatureSet<double> input);
}