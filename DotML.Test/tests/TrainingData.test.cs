using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class TrainingDataTest {

    private static TrainingSet Make(int count) {
        var generator = Enumerable.Range(0, count).Select(x => new TrainingPair { Input = new Vec<double>(), Output = new Vec<double>() });
        return new TrainingSet(generator);
    }

    [TestMethod]
    public void TestSampleSequentially() {
        var data = Make(100);
        var original = data.ToArray();

        var items = data.SampleSequentially().AsEnumerable().ToArray();

        Assert.AreEqual(100, data.Size);
        Assert.AreEqual(data.Size, original.Length);
        Assert.AreEqual(data.Size, items.Length);
        for (var i = 0; i < data.Size; i++) {
            Assert.IsTrue(ReferenceEquals(original[i], items[i]));
        }
    }

    [TestMethod]
    public void TestSampleRandomly() {
        var data = Make(100);
        var original = data.ToArray();

        var items = data.SampleRandomly().AsEnumerable().ToArray();

        Assert.AreEqual(100, data.Size);
        Assert.AreEqual(data.Size, original.Length);
        Assert.AreEqual(data.Size, items.Length);
    }

    [TestMethod]
    public void TestSampleRandomlySubset() {
        var data = Make(100);
        var original = data.ToArray();

        var items = data.SampleRandomly(50).AsEnumerable().ToArray();

        Assert.AreEqual(100, data.Size);
        Assert.AreEqual(50, items.Length);
    }

    [TestMethod]
    public void TestBatchSampleEvenly() {
        var data = Make(100);
        var original = data.ToArray();

        var items = data.SampleSequentially().AsBatchedEnumerable(10).ToList();

        Assert.AreEqual(100, data.Size);
        Assert.AreEqual(10, items.Count);
        var batch_index = 0;
        foreach (TrainingPairSequencer.TrainingPairBatch batch in items) {
            Assert.AreEqual(batch_index, batch.Key);
            Assert.AreEqual(10, batch.Size, $"Wrong batch size on batch {batch_index}.");
            Assert.AreEqual(10, batch.Count());
            batch_index++;
        }
    }
    [TestMethod]
    public void TestBatchSampleOdd() {
        var data = Make(105);
        var original = data.ToArray();

        var items = data.SampleSequentially().AsBatchedEnumerable(10).ToList();

        Assert.AreEqual(105, data.Size);
        Assert.AreEqual(11, items.Count);
        var batch_index = 0;
        foreach (TrainingPairSequencer.TrainingPairBatch batch in items) {
            if (batch_index != 10) {
                Assert.AreEqual(batch_index, batch.Key);
                Assert.AreEqual(10, batch.Size);
                Assert.AreEqual(10, batch.Count());
            } else {
                Assert.AreEqual(batch_index, batch.Key);
                Assert.AreEqual(5, batch.Size);
                Assert.AreEqual(5, batch.Count());
            }
            batch_index++;
        }
    }
}