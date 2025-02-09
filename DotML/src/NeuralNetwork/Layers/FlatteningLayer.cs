using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer which flattens inputs into a column vector (not-necessary as FullConnectedLayer will auto-flatten)
/// </summary>
[Untested()]
public class FlatteningLayer : FeedforwardNetworkLayer {
    public override void Initialize(IInitializer initializer) { }
    public override int TrainableParameterCount() => 0;

    public FlatteningLayer(Shape3D input_size) {
        this.InputShape = input_size;
        this.OutputShape = new Shape3D(1, input_size.Count, 1);
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> inputs) {
        // input is a 2D matrix processed from prior layers like a pooling layer
        var x = inputs.Channels == 1 && inputs[0].IsColumnMatrix ? inputs[0] : Matrix<double>.Column(inputs.SelectMany(x => x.FlattenRows()).ToArray());
        return new FeatureSet<double>(x);
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        FeatureSet<double>[] batched_input_errors = new FeatureSet<double>[args.OutputBatch.Batches];

        Parallel.For(0, args.OutputBatch.Batches, batchIndex => {
            var error = args.OutputErrors[batchIndex];
            var input = args.InputBatch[batchIndex];

            Matrix<double>[] input_errors;
            if (input.Channels == 1 && input.Shape == error.Shape) {
                input_errors = error.AsArray();                             // Same shape, no need to reshape
            } else {
                input_errors = error[0].Reshape(                            // Reshape to un-flatten error vector to match the input dimensions (in case next layer is not a fully connected layer)
                    input.Select(x => x.Shape))
                .ToArray();
            } 

            batched_input_errors[batchIndex] = new FeatureSet<double>(input_errors);
        });

        return new BackpropagationReturns(
            new BatchedFeatureSet<double>(batched_input_errors),
            null
        );
    }

    public override void SubtractGradients(LayerGradients? gradients) { }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(ILayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}