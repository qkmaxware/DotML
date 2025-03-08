using DotML;

public static class AssertExt {

    public static void AreEqual(BatchedFeatureSet<double> Y_truth, BatchedFeatureSet<double> Y_projected, double epsilon = 0.001) {
        Assert.AreEqual(Y_truth.Batches, Y_projected.Batches);
        var batch = 0;
        foreach (var (batch_predicted, batch_truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(batch_predicted.Channels, batch_truth.Channels);
            var channel = 0;
            foreach (var (channel_predicted, channel_truth) in batch_predicted.Zip(batch_truth)) {
                Assert.AreEqual(channel_truth.Rows, channel_predicted.Rows);
                Assert.AreEqual(channel_truth.Columns, channel_predicted.Columns);
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
}