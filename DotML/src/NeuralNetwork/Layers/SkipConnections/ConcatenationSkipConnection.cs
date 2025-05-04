using System.Diagnostics.CodeAnalysis;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public class ConcatenationSkipConnection : SkipConnection {

    public enum Side {
        ResidualLeft,
        ResidualRight
    }

    public Side ConcatenationSide {get; init;}
    public ConcatenationSkipConnection(Shape3D input_shape, InputCapture captureSource, Side side) 
        : base(input_shape, new Shape3D(input_shape.Channels + captureSource.OutputShape.Channels, input_shape.Rows, input_shape.Columns), captureSource) {
            this.ConcatenationSide = side;
        }

    /// <summary>
    /// Combine the features from the capture source with the features passed into this connection as input
    /// </summary>
    /// <param name="skipConnection">capture source features</param>
    /// <param name="input">input features</param>
    /// <returns>combined features</returns>
    public override BatchedFeatureSet<double> Combine(BatchedFeatureSet<double>? skipConnection, BatchedFeatureSet<double> inputs) {
        if (skipConnection is null)
            return inputs;

        if (skipConnection.Batches != inputs.Batches || skipConnection.Rows != inputs.Rows || skipConnection.Columns != inputs.Columns || (skipConnection.Channels + inputs.Channels) != inputs.Channels)
            throw new ArgumentException("Incompatible tensor shapes");

        var batches = skipConnection.Batches;
        var shape = this.OutputShape;

        // Loop over all channels and perform matrix addition
        FeatureSet<double>[] values = new FeatureSet<double>[batches];
        for (var batchIndex = 0; batchIndex < values.Length; batchIndex++) {
            var captured = skipConnection[batchIndex];
            var input = inputs[batchIndex];

            Matrix<double>[] features = new Matrix<double>[shape.Channels];
            var i = 0;
            if (ConcatenationSide == Side.ResidualRight) {
                // Residual to the Right
                var size_1 = inputs.Channels;
                var size_2 = features.Length;
                for (; i < size_1; i++) {
                    features[i] = input[i];
                }
                for (; i < size_2; i++) {
                    features[i] = captured[i - size_1];
                }
            } else {
                // Residual to the Left
                var size_1 = captured.Channels;
                var size_2 = features.Length;
                for (; i < size_1; i++) {
                    features[i] = captured[i];
                }
                for (; i < size_2; i++) {
                    features[i] = input[i - size_1];
                }
            }

            values[batchIndex] = new FeatureSet<double>(features);
        }

        return new BatchedFeatureSet<double>(values);
    }

    public override void Initialize(IInitializer initializer) { }

    public override int TrainableParameterCount() => 0;

    public override void SubtractGradients(LayerGradients? gradients) { }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        var residual_size = this.CaptureSource.OutputShape.Channels;
        var input_size = this.InputShape.Channels;

        if (ConcatenationSide == Side.ResidualLeft) {
            // Remove LHS channels
            var error_trim = new FeatureSet<double>[args.dY.Batches];
            for (var batch = 0; batch < args.dY.Batches; batch++) {
                var error_feats = new Matrix<double>[input_size];
                for (var i = 0; i < input_size; i++) {
                    error_feats[i] = args.dY[batch, i + residual_size];
                }
                error_trim[batch] = new FeatureSet<double>(error_feats);
            } 

            return new BackpropagationReturns (
                error: new BatchedFeatureSet<double>(error_trim), 
                gradients: null
            );
        } else {
            // Remove RHS channels
            var error_trim = new FeatureSet<double>[args.dY.Batches];
            for (var batch = 0; batch < args.dY.Batches; batch++) {
                var error_feats = new Matrix<double>[input_size];
                for (var i = 0; i < input_size; i++) {
                    error_feats[i] = args.dY[batch, i];
                }
                error_trim[batch] = new FeatureSet<double>(error_feats);
            } 

            return new BackpropagationReturns (
                error: new BatchedFeatureSet<double>(error_trim), 
                gradients: null
            );
        }
    }

    public override void Visit(ILayerVisitor visitor) => throw new NotImplementedException();// => visitor.Visit(this);
    
    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => throw new NotImplementedException();// => visitor.Visit(this, args);

    public override T Visit<T>(ILayerOutputVisitor<T> visitor) => throw new NotImplementedException();// => visitor.Visit(this);

    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => throw new NotImplementedException();// => visitor.Visit(this, args);
}