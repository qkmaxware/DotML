namespace DotML.Network;

[WorkInProgress]
public abstract class ResidualConnection : AdditionSkipConnection {
    public ResidualConnection(InputCapture captureSource) : base(captureSource) { }  

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
}