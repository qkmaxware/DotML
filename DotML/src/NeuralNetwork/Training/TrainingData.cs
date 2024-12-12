using System.Collections;

namespace DotML.Network.Training;

/// <summary>
/// A pair of inputs to matching outputs used in training of a neural network.
/// </summary>
public record class TrainingPair {
    /// <summary>
    /// Input vector to pass into the network
    /// </summary>
    public Vec<double> Input {get; set;}
    /// <summary>
    /// Output/classification produced by the input passing through the network
    /// </summary>
    public Vec<double> Output {get; set;}
}

/// <summary>
/// An enumerator that provides a way to access training data in a specific order
/// </summary>
public abstract class TrainingPairSequencer : IEnumerator<TrainingPair> {
    protected TrainingSet Datum {get; private set;}
    public virtual int Size => Datum.Size;

    public TrainingPairSequencer(TrainingSet datum) => Datum = datum;

    public abstract TrainingPair Current {get;}
    object IEnumerator.Current => this.Current;
    public void Dispose() {}
    public abstract bool MoveNext();
    public abstract void Reset();
}

/// <summary>
/// An enumerator that accesses training data in the order in which is was defined
/// </summary>
public class InOrderSequencer : TrainingPairSequencer {
    int current = -1;
    public InOrderSequencer(TrainingSet set) : base(set) { }

    public override TrainingPair Current => Datum[current];

    public override bool MoveNext() {
        if ((current + 1) < Datum.Size) {
            current += 1;
            return true;
        } else {
            return false;
        }
    }

    public override void Reset() {
        current = -1;
    }
}

public class RandomSequencer : TrainingPairSequencer {

    private int max_taken;
    private int current = -1;

    public RandomSequencer(TrainingSet set, int amount) : base(set) {
        this.max_taken = Math.Max(1, amount);
        this.current = -1;
    }

    private TrainingPair? selected = null;
    public override TrainingPair Current => selected is not null ? selected : throw new IndexOutOfRangeException();

    private Random rng = new Random();

    public override bool MoveNext() {
        if ((current + 1) < max_taken) {
            current += 1;
            if (Datum.Size > 0)
                selected = Datum[rng.Next(Datum.Size)];
            return true;
        } else {
            return false;
        }
    }

    public override void Reset() {
        this.selected = null;
        this.current = -1;
    }
}

/// <summary>
/// An enumerator that accesses the training data in a randomized order
/// </summary>
public class ShuffledSequencer : TrainingPairSequencer {
    private static readonly Random rng = new Random();
    private List<TrainingPair> shuffled;
    private int current = -1;

    public ShuffledSequencer(TrainingSet set) : base(set) {
        shuffled = [..set];
        shuffle();
        current = -1;
    }

    private void shuffle() {
        int n = shuffled.Count;

        // Fisher-Yates shuffle algorithm
        for (int i = n - 1; i > 0; i--) {
            // Generate a random index
            int j = rng.Next(0, i + 1);

            // Swap the elements
            var temp = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = temp;
        }
    }

    public override TrainingPair Current => shuffled[current];

    public override bool MoveNext() {
        if ((current + 1) < shuffled.Count) {
            current += 1;
            return true;
        } else {
            return false;
        }
    }

    public override void Reset() {
        shuffle();
        current = -1;
    }
}

/// <summary>
/// Description of a set containing network training data
/// </summary>
public interface ITrainingDataSet {
    public TrainingPairSequencer SampleSequentially();
    public TrainingPairSequencer SampleRandomly();
}

/// <summary>
/// A set of training data for training a neural network
/// </summary>
public class TrainingSet : IEnumerable<TrainingPair>, ITrainingDataSet {
    private List<TrainingPair> data {get; init;}

    public TrainingSet() {
        this.data = new List<TrainingPair>();
    }

    public TrainingSet(TrainingPair first, params TrainingPair[] next) {
        this.data = [first, ..next];
    }
    public TrainingSet(params IEnumerable<TrainingPair>[] datas) {
        this.data = new List<TrainingPair>();
        foreach (var d in datas)
            this.data.AddRange(d);
    }

    public TrainingSet(IEnumerable<TrainingPair> data) {
        this.data = [..data];
    }

    public void Add(TrainingPair pair) => data.Add(pair);
    public void Add(Vec<double> input, Vec<double> output) => data.Add(new TrainingPair{ Input = input, Output = output});
    public void AddRange(IEnumerable<TrainingPair> items) => data.AddRange(items);

    public TrainingPair this[int index] {
        get {
            if (index < 0 || index >= data.Count) {
                throw new IndexOutOfRangeException($"Index {index} is out of range of the dataset");
            }
            return this.data[index];
        }
    }
    public int Size => this.data.Count;

    /// <summary>
    /// Sample the training data in sequential order
    /// </summary>
    /// <returns>sequence</returns>
    public TrainingPairSequencer SampleSequentially() => new InOrderSequencer(this);

    /// <summary>
    /// Sample the training data in a random order
    /// </summary>
    /// <returns>sequence</returns>
    public TrainingPairSequencer SampleRandomly() => new ShuffledSequencer(this);

    /// <summary>
    /// Sample the training data in a random order a certain number of times
    /// </summary>
    /// <param name="count">number of samples to take</param>
    /// <returns>sequence</returns>
    public TrainingPairSequencer SampleRandomly(int count) => new RandomSequencer(this, count);

    public IEnumerator<TrainingPair> GetEnumerator() => this.data.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => this.data.GetEnumerator();

    /// <summary>
    /// Split the data into groups as evenly distributed as possible
    /// </summary>
    /// <param name="groupCount">number of groups</param>
    /// <returns>groups</returns>
    public IEnumerable<IEnumerable<TrainingPair>> SplitEvenly(int groupCount) {
        int totalCount = this.Size;
        int groupSize = (int)(Math.Ceiling((double)totalCount / (double)groupCount));

        int startIndex = 0;
        for (int i = 0; i < groupCount; i++) {
            yield return this.Skip(startIndex).Take(groupSize);
            startIndex += groupSize;
        }
    }
}