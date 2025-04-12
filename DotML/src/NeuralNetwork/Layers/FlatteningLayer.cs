using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer which reshapes the input into an new shape
/// </summary>
[Untested()]
public class ReshapeLayer : FeedforwardNetworkLayer {
    /// <summary>
    /// Create a new reshape layer
    /// </summary>
    /// <param name="input_shape">The shape of allowed input</param>
    /// <param name="output_shape">The resulting shape of the input after reshaping</param>
    public ReshapeLayer(Shape3D input_shape, Shape3D output_shape) {
        this.InputShape = input_shape;
        this.OutputShape = output_shape;
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        if (channels.Shape != OutputShape)
            return channels.Reshape(OutputShape);
        return channels;
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        FeatureSet<double>[] batched_input_errors = new FeatureSet<double>[args.OutputBatch.Batches];

        Parallel.For(0, args.OutputBatch.Batches, batchIndex => {
            var error = args.OutputErrors[batchIndex];
            var input = args.InputBatch[batchIndex];

            FeatureSet<double> input_errors;
            if (input.Shape == error.Shape) {
                input_errors = error;                             // Same shape, no need to reshape
            } else {
                input_errors = error.Reshape(this.InputShape);    // Reshape the output to match the input shape
            } 

            batched_input_errors[batchIndex] = input_errors;
        });

        return new BackpropagationReturns(
            new BatchedFeatureSet<double>(batched_input_errors),
            null
        );
    }

    public override void Initialize(IInitializer initializer) { }

    public override void SubtractGradients(LayerGradients? gradients) { }

    public override int TrainableParameterCount() => 0;

    public override void Visit(ILayerVisitor visitor) {
        throw new NotImplementedException();
    }

    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) {
        throw new NotImplementedException();
    }

    public override T Visit<T>(ILayerOutputVisitor<T> visitor) {
        throw new NotImplementedException();
    }

    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) {
        throw new NotImplementedException();
    }
}

/// <summary>
/// Layer which flattens inputs into a column vector (not-necessary as FullConnectedLayer will auto-flatten)
/// </summary>
[Untested()]
public class FlatteningLayer : ReshapeLayer {
    public FlatteningLayer(Shape3D input_size) : base(input_size, new Shape3D(1, input_size.Count, 1)) { }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FeatureSet<double> Flatten(FeatureSet<double> channels) {
        if (channels.Channels == 1 && channels[0].IsColumnMatrix) {
            return channels;
        } else {
            Matrix<double> output = new Matrix<double>(channels.Shape.Count, 1, channels.SelectMany(x => x.FlattenRows()));
            return new FeatureSet<double>(output);
        }
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> inputs) {
        // Input is a 2D matrix processed from prior layers like a pooling layer
        // If the input is already flattened, use that; otherwise, flatten the input.
        return Flatten(inputs);
    }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => visitor.Visit(this, args);
    public override T Visit<T>(ILayerOutputVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}