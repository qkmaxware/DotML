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

    /// <summary>
    /// Create a skip connection block
    /// <example>
    /// <code>
    /// .Then(block_input => SkipConnection.Block((output_shape, input_src) => new ResidualConnection(output_shape, input_src), [
    ///     new ConvolutionLayer(block_input),
    ///     ...
    /// ]))
    /// </code>
    /// </example>
    /// While this method can be used to create skip connection blocks, it is a little unweildy and can result in a loss of clarity/semantics. It is recommended to use other methods to create specific skip connections such as:
    /// <list type="bullet">
    /// <item>
    ///     <term>SkipConnection.ResidualBlock</term>
    ///     <description>Create a residual block</description>
    /// </item>
    /// </list>
    /// </summary>
    /// <param name="type">Skip connection type</param>
    /// <param name="sequencer">Layers in the block</param>
    /// <returns>Generator to create all layers</returns>
    public static NetworkBlockGenerator Block(Func<Shape3D, InputCapture, SkipConnection> type, IEnumerable<IFeedforwardNetworkLayer> sequencer)  {
        return (ishape) => {
            return MakeBlockEnumerable(ishape, type, sequencer);
        };
    }

    /// <summary>
    /// Create a residual connection block 
    /// <example>
    /// <code>
    /// .Then(block_input => SkipConnection.ResidualBlock([
    ///     new ConvolutionLayer(block_input),
    ///     ...
    /// ]))
    /// </code>
    /// </example>
    /// </summary>
    /// <param name="sequencer">Layers in the block</param>
    /// <returns>Generator to create all layers</returns>
    public static NetworkBlockGenerator ResidualBlock(IEnumerable<IFeedforwardNetworkLayer> sequencer) => Block((oshape, input) => new ResidualConnection(oshape, input), sequencer);

    private static IEnumerable<IFeedforwardNetworkLayer> MakeBlockEnumerable(Shape3D input_shape, Func<Shape3D, InputCapture, SkipConnection> type, IEnumerable<IFeedforwardNetworkLayer> sequencer) {
        // Bookend the layers with:
        var first = new InputCapture(input_shape);            // An input capture to pass onto the end of the skip connection
        var shape = first.OutputShape;
        yield return first;
        foreach (var layer in sequencer) {
            yield return layer;
            shape = layer.OutputShape;
        }
        var last = type(shape, first);    // The actual end of the skip connection
        yield return last;
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