using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public abstract class AdditionSkipConnection : SkipConnection {
    public AdditionSkipConnection(Shape3D input_shape, InputCapture captureSource) : base(input_shape, input_shape, captureSource) { }

    /// <summary>
    /// Combine the features from the capture source with the features passed into this connection as input
    /// </summary>
    /// <param name="skipConnection">capture source features</param>
    /// <param name="input">input features</param>
    /// <returns>combined features</returns>
    public override BatchedFeatureSet<double> Combine(BatchedFeatureSet<double>? skipConnection, BatchedFeatureSet<double> input) {
        if (skipConnection is null)
            return input;

        if (skipConnection.Shape != input.Shape)
            throw new ArgumentException("Incompatible tensor shapes");

        var shape = input.Shape;

        // Loop over all channels and perform matrix addition
        FeatureSet<double>[] values = new FeatureSet<double>[shape.Batches];
        for (var batchIndex = 0; batchIndex < values.Length; batchIndex++) {
            Matrix<double>[] features = new Matrix<double>[shape.Channels];

            for (var i = 0; i < features.Length; i++) {
                features[i] = skipConnection[batchIndex][i] + input[batchIndex][i];
            }

            values[batchIndex] = new FeatureSet<double>(features);
        }

        return new BatchedFeatureSet<double>(values);
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
}