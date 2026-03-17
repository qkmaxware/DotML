using DotML.Network.Training;

namespace DotML.Test.Training;

[TestClass]
public class TestSamplers
{   
    private static ITrainingDataSource<double> MakeSrc(int count) {
        var generator = Enumerable.Range(0, count).Select(x => (Tensor<double>.Vec([(double)x]), Tensor<double>.Vec([(double)x])) );
        ListTrainingDataSource<double> src = new(new Shape(0), new Shape(0));
        foreach (var pair in generator)
            src.Add(pair);
        return src;
    }

    [TestMethod]
    public void TestSequentialBatch1()
    {
        // Init test tensors (predictable)
        var ishape = new Shape(4, 3);
        var oshape = new Shape(4, 3);
        ListTrainingDataSource<int> values = new ListTrainingDataSource<int>(ishape, oshape);
        for (var i = 0; i < 10; i++)
        {
            Tensor<int> input = Tensor<int>.ConstantValued(ishape, i);
            Tensor<int> output = Tensor<int>.ConstantValued(oshape, i);
            values.Add((input, output));
        }

        // Test sampler
        var sampler = ((ITrainingDataSource<int>)values).CreateSequentialSampler();

        var index = 0;
        foreach (var (i, o) in sampler.Sample(batchSize: 1))
        {
            // See that the batch dimension got added
            Assert.AreEqual(ishape.Rank + 1, i.Rank);
            Assert.AreEqual(oshape.Rank + 1, o.Rank);
            Assert.AreEqual(1, i.Shape[0]);

            var pair = values[index];
            Assert.IsTrue(i.SubtensorSpan(0).SequenceEqual(pair.Input.AsSpan()));
            Assert.IsTrue(o.SubtensorSpan(0).SequenceEqual(pair.Output.AsSpan()));

            index++;
        }
    }

    [TestMethod]
    public void TestSequentialBatch2()
    {
        // Init test tensors (predictable)
        var ishape = new Shape(4, 3);
        var oshape = new Shape(4, 3);
        ListTrainingDataSource<int> values = new ListTrainingDataSource<int>(ishape, oshape);
        for (var i = 0; i < 10; i++)
        {
            Tensor<int> input = Tensor<int>.ConstantValued(ishape, i);
            Tensor<int> output = Tensor<int>.ConstantValued(oshape, i);
            values.Add((input, output));
        }

        // Test sampler
        var sampler = ((ITrainingDataSource<int>)values).CreateSequentialSampler();

        var index = 0;
        foreach (var (i, o) in sampler.Sample(batchSize: 2))
        {
            // See that the batch dimension got added
            Assert.AreEqual(ishape.Rank + 1, i.Rank);
            Assert.AreEqual(oshape.Rank + 1, o.Rank);
            Assert.AreEqual(2, i.Shape[0]);

            var pair = values[index];
            Assert.IsTrue(i.SubtensorSpan(0).SequenceEqual(pair.Input.AsSpan()));
            Assert.IsTrue(o.SubtensorSpan(0).SequenceEqual(pair.Output.AsSpan()));
            index++;

            pair = values[index];
            Assert.IsTrue(i.SubtensorSpan(1).SequenceEqual(pair.Input.AsSpan()));
            Assert.IsTrue(o.SubtensorSpan(1).SequenceEqual(pair.Output.AsSpan()));
            index++;
        }
    }
    
    [TestMethod]
    public void TestRandomNoDupBatch1()
    {
        // Init test tensors (predictable)
        var ishape = new Shape(4, 3);
        var oshape = new Shape(4, 3);
        ListTrainingDataSource<int> values = new ListTrainingDataSource<int>(ishape, oshape);
        HashSet<int> unfound = new HashSet<int>();
        for (var i = 0; i < 10; i++)
        {
            Tensor<int> input = Tensor<int>.ConstantValued(ishape, i);
            Tensor<int> output = Tensor<int>.ConstantValued(oshape, i);
            values.Add((input, output));
            unfound.Add(i);
        }

        // Test sampler
        var sampler = ((ITrainingDataSource<int>)values).CreateRandomSampler(allowDuplicates: false);

        foreach (var (i, o) in sampler.Sample(batchSize: 1))
        {
            // See that the batch dimension got added
            Assert.AreEqual(ishape.Rank + 1, i.Rank);
            Assert.AreEqual(oshape.Rank + 1, o.Rank);
            Assert.AreEqual(1, i.Shape[0]);

            var index = i.SubtensorSpan(0)[0];
            unfound.Remove(index);
            var pair = values[index];
            Assert.IsTrue(i.SubtensorSpan(0).SequenceEqual(pair.Input.AsSpan()));
            Assert.IsTrue(o.SubtensorSpan(0).SequenceEqual(pair.Output.AsSpan()));
            index++;
        }

        Assert.IsTrue(unfound.Count == 0);
    }

    [TestMethod]
    public void TestRandomNoDupBatch2()
    {
        // Init test tensors (predictable)
        var ishape = new Shape(4, 3);
        var oshape = new Shape(4, 3);
        ListTrainingDataSource<int> values = new ListTrainingDataSource<int>(ishape, oshape);
        HashSet<int> unfound = new HashSet<int>();
        for (var i = 0; i < 10; i++)
        {
            Tensor<int> input = Tensor<int>.ConstantValued(ishape, i);
            Tensor<int> output = Tensor<int>.ConstantValued(oshape, i);
            values.Add((input, output));
            unfound.Add(i);
        }

        // Test sampler
        var sampler = ((ITrainingDataSource<int>)values).CreateRandomSampler(allowDuplicates: false);

        foreach (var (i, o) in sampler.Sample(batchSize: 2))
        {
            // See that the batch dimension got added
            Assert.AreEqual(ishape.Rank + 1, i.Rank);
            Assert.AreEqual(oshape.Rank + 1, o.Rank);
            Assert.AreEqual(2, i.Shape[0]);

            var index = i.SubtensorSpan(0)[0];
            if (!unfound.Contains(index))
                Assert.Fail("Duplicate found");
            unfound.Remove(index);
            var pair = values[index];
            Assert.IsTrue(i.SubtensorSpan(0).SequenceEqual(pair.Input.AsSpan()));
            Assert.IsTrue(o.SubtensorSpan(0).SequenceEqual(pair.Output.AsSpan()));
            index++;

            index = i.SubtensorSpan(1)[0];
            if (!unfound.Contains(index))
                Assert.Fail("Duplicate found");
            unfound.Remove(index);
            pair = values[index];
            Assert.IsTrue(i.SubtensorSpan(1).SequenceEqual(pair.Input.AsSpan()));
            Assert.IsTrue(o.SubtensorSpan(1).SequenceEqual(pair.Output.AsSpan()));
            index++;
        }

        Assert.IsTrue(unfound.Count == 0);
    }

    [TestMethod]
    public void TestRandomSubsetNoDupBatch1()
    {
        const int SIZE = 100;

        // Init test tensors (predictable)
        var ishape = new Shape(4, 3);
        var oshape = new Shape(4, 3);
        ListTrainingDataSource<int> values = new ListTrainingDataSource<int>(ishape, oshape);
        HashSet<int> unfound = new HashSet<int>(SIZE);
        for (var i = 0; i < SIZE; i++)
        {
            Tensor<int> input = Tensor<int>.ConstantValued(ishape, i);
            Tensor<int> output = Tensor<int>.ConstantValued(oshape, i);
            values.Add((input, output));
            unfound.Add(i);
        }
        Assert.AreEqual(SIZE, values.Count);

        // Test sampler
        const int SUBSET_SIZE = 50;
        var sampler = ((ITrainingDataSource<int>)values).CreateRandomSampler(SUBSET_SIZE, allowDuplicates: false);
        Assert.AreEqual(SUBSET_SIZE, sampler.Count);

        foreach (var (i, o) in sampler.Sample(batchSize: 1))
        {
            // See that the batch dimension got added
            Assert.AreEqual(ishape.Rank + 1, i.Rank);
            Assert.AreEqual(oshape.Rank + 1, o.Rank);
            Assert.AreEqual(1, i.Shape[0]);

            var index = i[0];
            unfound.Remove(index);
            var pair = values[index];
            Assert.IsTrue(i.SubtensorSpan(0).SequenceEqual(pair.Input.AsSpan()));
            Assert.IsTrue(o.SubtensorSpan(0).SequenceEqual(pair.Output.AsSpan()));
            index++;
        }

        Assert.AreEqual(SIZE - SUBSET_SIZE, unfound.Count);
    }

    [TestMethod]
    public void TestRandomSubsetNoDupBatch2()
    {
        const int SIZE = 100;

        // Init test tensors (predictable)
        var ishape = new Shape(4, 3);
        var oshape = new Shape(4, 3);
        ListTrainingDataSource<int> values = new ListTrainingDataSource<int>(ishape, oshape);
        HashSet<int> unfound = new HashSet<int>();
        for (var i = 0; i < SIZE; i++)
        {
            Tensor<int> input = Tensor<int>.ConstantValued(ishape, i);
            Tensor<int> output = Tensor<int>.ConstantValued(oshape, i);
            values.Add((input, output));
            unfound.Add(i);
        }
        Assert.AreEqual(SIZE, values.Count);

        // Test sampler
        const int SUBSET_SIZE = 50;
        var sampler = ((ITrainingDataSource<int>)values).CreateRandomSampler(SUBSET_SIZE, allowDuplicates: false);
        Assert.AreEqual(SUBSET_SIZE, sampler.Count);

        foreach (var (i, o) in sampler.Sample(batchSize: 2))
        {
            // See that the batch dimension got added
            Assert.AreEqual(ishape.Rank + 1, i.Rank);
            Assert.AreEqual(oshape.Rank + 1, o.Rank);
            Assert.AreEqual(2, i.Shape[0]);

            var index = i.SubtensorSpan(0)[0];
            if (!unfound.Contains(index))
                Assert.Fail("Duplicate found");
            unfound.Remove(index);
            var pair = values[index];
            Assert.IsTrue(i.SubtensorSpan(0).SequenceEqual(pair.Input.AsSpan()));
            Assert.IsTrue(o.SubtensorSpan(0).SequenceEqual(pair.Output.AsSpan()));
            index++;

            index = i.SubtensorSpan(1)[0];
            if (!unfound.Contains(index))
                Assert.Fail("Duplicate found");
            unfound.Remove(index);
            pair = values[index];
            Assert.IsTrue(i.SubtensorSpan(1).SequenceEqual(pair.Input.AsSpan()));
            Assert.IsTrue(o.SubtensorSpan(1).SequenceEqual(pair.Output.AsSpan()));
            index++;
        }

        Assert.AreEqual(SIZE - SUBSET_SIZE, unfound.Count);
    }
}
