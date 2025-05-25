using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public class AdditionSkipConnection : SkipConnection {
    public AdditionSkipConnection(Shape3D input_shape, InputCapture captureSource) : base(input_shape, input_shape, captureSource) { }

    /// <summary>
    /// Combine the features from the capture source with the features passed into this connection as input
    /// </summary>
    /// <param name="skipConnection">capture source features</param>
    /// <param name="input">input features</param>
    /// <returns>combined features</returns>
    public override BatchedFeatureSet<float> Combine(BatchedFeatureSet<float>? skipConnection, BatchedFeatureSet<float> inputs) {
        if (skipConnection is null)
            return inputs;

        if (skipConnection.Shape != inputs.Shape)
            throw new ArgumentException("Incompatible tensor shapes");

        var shape = inputs.Shape;

        // Loop over all channels and perform matrix addition
        FeatureSet<float>[] values = new FeatureSet<float>[shape.Batches];
        for (var batchIndex = 0; batchIndex < values.Length; batchIndex++) {
            Matrix<float>[] features = new Matrix<float>[shape.Channels];
            var captured = skipConnection[batchIndex];
            var input = inputs[batchIndex];

            for (var i = 0; i < features.Length; i++) {
                features[i] = captured[i] + input[i];
            }

            values[batchIndex] = new FeatureSet<float>(features);
        }

        return new BatchedFeatureSet<float>(values);
    }

    public override void Initialize(IInitializer initializer) { }

    public override int TrainableParameterCount() => 0;

    public override void SubtractGradients(LayerGradients? gradients) { }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        // Loss is a function of L(y(residual, x))
        // dL/dx = dL/dy * dy/dx
        // Soooo
        // remember y(residual, x) := x + residual
        // dL/dy = given
        // dy/dx = d/dx(x) + d/dx(residual) => 1 + 0 => 1 where residual is considered a constant here
        // dL/dx = dL/dy * 1

        return new BackpropagationReturns (
            error: args.dY, 
            gradients: null
        );
    }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    
    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => visitor.Visit(this, args);

    public override T Visit<T>(ILayerOutputVisitor<T> visitor) => visitor.Visit(this);

    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}