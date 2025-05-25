using DotML;

public static class AssertExt {

    public static void AreEqual(Vec<double> y_truth, Vec<double> y_projected, double epsilon = 0.001) {
        Assert.AreEqual(y_truth.Dimensionality, y_projected.Dimensionality);
        for (var i = 0; i < y_truth.Dimensionality; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }

    public static void AreEqual(Matrix<double> y_truth, Matrix<double> y_projected, double epsilon = 0.001) {
        Assert.AreEqual(y_truth.Size, y_projected.Size);
        for (var i = 0; i < y_truth.Size; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }

    public static void AreEqual(BatchedFeatureSet<double> Y_truth, BatchedFeatureSet<double> Y_projected, double epsilon = 0.001) {
        Assert.AreEqual(Y_truth.Batches, Y_projected.Batches, "Batch count isn't equal");
        var batch = 0;
        foreach (var (batch_predicted, batch_truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(batch_predicted.Channels, batch_truth.Channels, "Channel count isn't equal");
            var channel = 0;
            foreach (var (channel_predicted, channel_truth) in batch_predicted.Zip(batch_truth)) {
                Assert.AreEqual(channel_truth.Rows, channel_predicted.Rows, "Row count isn't equal");
                Assert.AreEqual(channel_truth.Columns, channel_predicted.Columns, "Column count isn't equal");
                var index = 0;
                foreach (var (predicted, truth) in channel_predicted.Zip(channel_truth)) {
                    Assert.AreEqual(truth, predicted, epsilon, $"Element {batch}x{channel}x{index}");
                    index++;
                } 
                channel++;   
            }
            batch++;
        }
    }

    public static void AreEqual(Vec<float> y_truth, Vec<float> y_projected, float epsilon = 0.001f) {
        Assert.AreEqual(y_truth.Dimensionality, y_projected.Dimensionality);
        for (var i = 0; i < y_truth.Dimensionality; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }

    public static void AreEqual(Matrix<float> y_truth, Matrix<float> y_projected, float epsilon = 0.001f) {
        Assert.AreEqual(y_truth.Size, y_projected.Size);
        for (var i = 0; i < y_truth.Size; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }

    public static void AreEqual(BatchedFeatureSet<float> Y_truth, BatchedFeatureSet<float> Y_projected, float epsilon = 0.001f) {
        Assert.AreEqual(Y_truth.Batches, Y_projected.Batches, "Batch count isn't equal");
        var batch = 0;
        foreach (var (batch_predicted, batch_truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(batch_predicted.Channels, batch_truth.Channels, "Channel count isn't equal");
            var channel = 0;
            foreach (var (channel_predicted, channel_truth) in batch_predicted.Zip(batch_truth)) {
                Assert.AreEqual(channel_truth.Rows, channel_predicted.Rows, "Row count isn't equal");
                Assert.AreEqual(channel_truth.Columns, channel_predicted.Columns, "Column count isn't equal");
                var index = 0;
                foreach (var (predicted, truth) in channel_predicted.Zip(channel_truth)) {
                    Assert.AreEqual(truth, predicted, epsilon, $"Element {batch}x{channel}x{index}");
                    index++;
                } 
                channel++;   
            }
            batch++;
        }
    }

    public static void AreEqual(Vec<double> y_truth, Vec<float> y_projected, float epsilon = 0.001f) {
        Assert.AreEqual(y_truth.Dimensionality, y_projected.Dimensionality);
        for (var i = 0; i < y_truth.Dimensionality; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }

    public static void AreEqual(Matrix<double> y_truth, Matrix<float> y_projected, float epsilon = 0.001f) {
        Assert.AreEqual(y_truth.Size, y_projected.Size);
        for (var i = 0; i < y_truth.Size; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }

    public static void AreEqual(BatchedFeatureSet<double> Y_truth, BatchedFeatureSet<float> Y_projected, float epsilon = 0.001f) {
        Assert.AreEqual(Y_truth.Batches, Y_projected.Batches, "Batch count isn't equal");
        var batch = 0;
        foreach (var (batch_predicted, batch_truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(batch_predicted.Channels, batch_truth.Channels, "Channel count isn't equal");
            var channel = 0;
            foreach (var (channel_predicted, channel_truth) in batch_predicted.Zip(batch_truth)) {
                Assert.AreEqual(channel_truth.Rows, channel_predicted.Rows, "Row count isn't equal");
                Assert.AreEqual(channel_truth.Columns, channel_predicted.Columns, "Column count isn't equal");
                var index = 0;
                foreach (var (predicted, truth) in channel_predicted.Zip(channel_truth)) {
                    Assert.AreEqual(truth, predicted, epsilon, $"Element {batch}x{channel}x{index}");
                    index++;
                } 
                channel++;   
            }
            batch++;
        }
    }

    public static void DumpAndBail(params BatchedFeatureSet<double>[] dumped) {
        throw new AggregateException(dumped.Select(x => new Exception(x.ToJaggedArrayString())));
    }
}