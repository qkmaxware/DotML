using System.Text.RegularExpressions;
using DotML.Network.Initialization;

namespace DotML.Network;

[WorkInProgress]
public abstract class SkipConnection : FeedforwardNetworkLayer {

    /// <summary>
    /// The capture source where the skip connection draws it's features from
    /// </summary>
    public InputCapture CaptureSource {get; private set;}

    public SkipConnection(InputCapture captureSource) {
        this.CaptureSource = captureSource;
    }

    /// <summary>
    /// <para>Create a skip connection block</para>
    /// <example>
    /// How to create a residual block:
    /// <code>
    /// .Then(block_input => SkipConnection.Block((block_output) => new ResidualConnection(block_output), 
    ///     new ConvolutionLayer(block_input),
    ///     new LocalMaxPoolingLayer(...),
    ///     new ActivationLayer(...)
    /// ))
    /// </code>
    /// </example>
    /// </summary>
    /// <param name="type">Skip connection type</param>
    /// <param name="layers">Layers in the block</param>
    /// <returns>Generator to create all layers</returns>
    public static NetworkBlockGenerator Block(Func<Shape3D, SkipConnection> type, params IFeedforwardNetworkLayer[] layers) {
        return (ishape) => {
            // Bookend the layers with:
            var first = new InputCapture();            // An input capture to pass onto the end of the skip connection
            var last = type(layers[^1].OutputShape);   // The actual end of the skip connection
            return layers.Prepend(first).Append(last);
        };
    }

    /// <summary>
    /// <para>Create a skip connection block</para>
    /// <example>
    /// How to create a residual block:
    /// <code>
    /// .Then(block_input => SkipConnection.Block((block_output) => new ResidualConnection(block_output), 
    ///     new ConvolutionLayer(block_input)
    ///     .Then(ishape => ...)
    ///     .Then(ishape => ...)
    ///     ...
    /// ))
    /// </code>
    /// </example>
    /// </summary>
    /// <param name="type">Skip connection type</param>
    /// <param name="sequencer">Layers in the block</param>
    /// <returns>Generator to create all layers</returns>
    public static NetworkBlockGenerator Block(Func<Shape3D, SkipConnection> type, LayerSequencer sequencer)  {
        return (ishape) => {
            // Bookend the layers with:
            var first = new InputCapture();            // An input capture to pass onto the end of the skip connection
            var last = type(sequencer.OutputShape);    // The actual end of the skip connection
            return sequencer.Prepend(first).Append(last);
        };
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