using System.Collections;
using System.Collections.ObjectModel;
using System.Dynamic;
using System.Numerics;
using System.Text.RegularExpressions;
using DotML.Network.Training;

namespace DotML.Network.Training;

/// <summary>
/// A pair of inputs to matching outputs used in training of a neural network.
/// </summary>
public record class TrainingPair<T> where T:INumber<T> {
    /// <summary>
    /// Input vector to pass into the network
    /// </summary>
    public Vec<T> Input {get; set;}
    /// <summary>
    /// Output/classification produced by the input passing through the network
    /// </summary>
    public Vec<T> Output {get; set;}
}

/// <summary>
/// An enumerator that provides a way to access training data in a specific order
/// </summary>
public abstract class TrainingPairSequencer<T> : IEnumerator<TrainingPair<T>> where T:INumber<T> {
    protected TrainingSet<T> Datum {get; private set;}
    public virtual int Size => Datum.Size;

    public TrainingPairSequencer(TrainingSet<T> datum) => Datum = datum;

    public abstract TrainingPair<T> Current {get;}
    object IEnumerator.Current => this.Current;
    public void Dispose() {}
    public abstract bool MoveNext();
    public abstract void Reset();

    /// <summary>
    /// Treat this enumerator as it's own enumerable object for use in for loops
    /// </summary>
    /// <returns>enumerable of training pairs</returns>
    public IEnumerable<TrainingPair<T>> AsEnumerable() {
        Reset();
        while (MoveNext())
            yield return Current;
    }

    /// <summary>
    /// Treat this enumerator as it's own enumerable object where elements are sampled in a batch
    /// </summary>
    /// <returns>enumerable of batches/groups of training pairs</returns>
    public IEnumerable<IGrouping<int, TrainingPair<T>>> AsBatchedEnumerable(int batch_size) {
        batch_size = Math.Max(1, batch_size);
        var batch = new List<TrainingPair<T>>(batch_size);

        // Create initial batch
        Reset();
        while (batch.Count < batch_size && MoveNext()) {
            batch.Add(this.Current);
        }

        int batch_index = 0;
        while (batch.Count > 0) {
            yield return new TrainingPairBatch(batch_index, batch);
            batch_index++;

            // Create subsequent batch
            batch = new List<TrainingPair<T>>(batch_size);
            while (batch.Count < batch_size && MoveNext()) {
                batch.Add(this.Current);
            }
        }
    }   

    /// <summary>
    /// Represents a group of training pairs of a given batch size
    /// </summary>
    public class TrainingPairBatch : IGrouping<int, TrainingPair<T>> {
        /// <summary>
        /// Batch number / Group key (0-indexed)
        /// </summary>
        public int Key {get; private set;}
        
        /// <summary>
        /// Batch size (number of elements)
        /// </summary>
        public int Size => items.Count();
        private IEnumerable<TrainingPair<T>> items;

        public TrainingPairBatch(int batch_index, IEnumerable<TrainingPair<T>> items) {
            this.Key = batch_index;
            this.items = items;
        }

        public IEnumerator<TrainingPair<T>> GetEnumerator() => items.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => items.GetEnumerator();
    }
}

/// <summary>
/// An enumerator that accesses training data in the order in which is was defined
/// </summary>
public class InOrderSequencer<T> : TrainingPairSequencer<T> where T:INumber<T> {
    int current = -1;
    public InOrderSequencer(TrainingSet<T> set) : base(set) { }

    public override TrainingPair<T> Current => Datum[current];

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

public class RandomSequencer<T> : TrainingPairSequencer<T> where T:INumber<T> {

    private int max_taken;
    private int current = -1;

    public RandomSequencer(TrainingSet<T> set, int amount) : base(set) {
        this.max_taken = Math.Max(1, amount);
        this.current = -1;
    }

    private TrainingPair<T>? selected = null;
    public override TrainingPair<T> Current => selected is not null ? selected : throw new IndexOutOfRangeException();

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
public class ShuffledSequencer<T> : TrainingPairSequencer<T> where T:INumber<T> {
    private static readonly Random rng = new Random();
    private List<TrainingPair<T>> shuffled;
    private int current = -1;

    public ShuffledSequencer(TrainingSet<T> set) : base(set) {
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

    public override TrainingPair<T> Current => shuffled[current];

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
public interface ITrainingDataSet<T> where T:INumber<T> {
    public TrainingPairSequencer<T> SampleSequentially();
    public TrainingPairSequencer<T> SampleRandomly();
}

/// <summary>
/// A set of training data for training a neural network
/// </summary>
public class TrainingSet<T> : IEnumerable<TrainingPair<T>>, ITrainingDataSet<T> where T:INumber<T> {
    private List<TrainingPair<T>> data {get; init;}

    public TrainingSet() {
        this.data = new List<TrainingPair<T>>();
    }

    public TrainingSet(int initial_capacity) {
        this.data = new List<TrainingPair<T>>(initial_capacity);
    }

    public TrainingSet(TrainingPair<T> first, params TrainingPair<T>[] next) {
        this.data = [first, ..next];
    }
    public TrainingSet(params IEnumerable<TrainingPair<T>>[] datas) {
        this.data = new List<TrainingPair<T>>();
        foreach (var d in datas)
            this.data.AddRange(d);
    }

    public TrainingSet(IEnumerable<TrainingPair<T>> data) {
        this.data = [..data];
    }

    /// <summary>
    /// Add a pair to the training set
    /// </summary>
    /// <param name="pair">training pair</param>
    public void Add(TrainingPair<T> pair) => data.Add(pair);

    /// <summary>
    /// Add an input/output pair to the training set
    /// </summary>
    /// <param name="input">training pair input</param>
    /// <param name="output">training pair output</param>
    public void Add(Vec<T> input, Vec<T> output) => data.Add(new TrainingPair<T>{ Input = input, Output = output});
    
    /// <summary>
    /// Add a range of training pairs to the training set
    /// </summary>
    /// <param name="items">training pairs</param>
    public void AddRange(IEnumerable<TrainingPair<T>> items) => data.AddRange(items);

    /// <summary>
    /// Get a particular training pair from the set
    /// </summary>
    /// <param name="index">index of the pair</param>
    /// <returns>training pair</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when the index is out of range</exception>
    public TrainingPair<T> this[int index] {
        get {
            if (index < 0 || index >= data.Count) {
                throw new IndexOutOfRangeException($"Index {index} is out of range of the dataset");
            }
            return this.data[index];
        }
    }

    /// <summary>
    /// Number of elements in the training set
    /// </summary>
    public int Size => this.data.Count;

    /// <summary>
    /// Sample the training data in sequential order
    /// </summary>
    /// <returns>sequence</returns>
    public TrainingPairSequencer<T> SampleSequentially() => new InOrderSequencer<T>(this);

    /// <summary>
    /// Sample the training data in a random order
    /// </summary>
    /// <returns>sequence</returns>
    public TrainingPairSequencer<T> SampleRandomly() => new ShuffledSequencer<T>(this);

    /// <summary>
    /// Sample the training data in a random order a certain number of times
    /// </summary>
    /// <param name="count">number of samples to take</param>
    /// <returns>sequence</returns>
    public TrainingPairSequencer<T> SampleRandomly(int count) => new RandomSequencer<T>(this, count);

    public IEnumerator<TrainingPair<T>> GetEnumerator() => this.data.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => this.data.GetEnumerator();

    /// <summary>
    /// Split the data into groups as evenly distributed as possible
    /// </summary>
    /// <param name="groupCount">number of groups</param>
    /// <returns>group of training sets</returns>
    public IEnumerable<TrainingSet<T>> SplitEvenly(int groupCount) {
        int totalCount = this.Size;
        int groupSize = (int)(Math.Ceiling((double)totalCount / (double)groupCount));

        int startIndex = 0;
        for (int i = 0; i < groupCount; i++) {
            yield return new TrainingSet<T>(this.Skip(startIndex).Take(groupSize));
            startIndex += groupSize;
        }
    }

    /// <summary>
    /// Split the data into multiple groups with elements from the group being determined by the conditions
    /// </summary>
    /// <param name="conditions">group conditions</param>
    /// <returns>group of training sets</returns>
    public IEnumerable<TrainingSet<T>> SplitWhen(params Predicate<(TrainingSet<T> Set, TrainingPair<T> Value)>[] conditions) {
        TrainingSet<T>[] sets = new TrainingSet<T>[conditions.Length];
        for(var i = 0; i < sets.Length; i++)
            sets[i] = new TrainingSet<T>();

        foreach (var pair in this.data) {
            for (var i = 0; i < conditions.Length; i++) {
                var condition = conditions[i];
                var set = sets[i];
                if (condition((set, pair))) {
                    set.Add(pair);
                    break;
                }
            }
        }
        
        foreach (var set in sets)
            yield return set;
    }

    private static Random rng = new Random();
    /// <summary>
    /// Split the data into multiple groups based on a flex system. 
    /// Flex 1, 1 would be a 50/50 split because each would contain 1 element out of a total span of 1+1 = 2 elements.
    /// 
    /// Common flex values: 
    /// - (1, 1) = 50%/50%
    /// - (1, 2) = 33%/66%
    /// - (1, 3) = 25%/75%
    /// </summary>
    /// <param name="flex">list of flex probabilities</param>
    /// <returns>group of training sets</returns>
    public IEnumerable<TrainingSet<T>> SplitProbabilistically(params int[] flex) {
        for (var i = 0; i < flex.Length; i++) {
            if (flex[i] < 0)
                throw new ArgumentException("Flex values must be greater than or equal to 0");
        }

        TrainingSet<T>[] sets = new TrainingSet<T>[flex.Length];
        for(var i = 0; i < sets.Length; i++)
            sets[i] = new TrainingSet<T>();

        var sum = flex.Sum();

        foreach (var pair in this.data) {
            var r = rng.NextDouble();
            var current_probability = 0.0;
            var set_index = 0;

            foreach (var ratio in flex) {
                current_probability += (double)ratio / (double)sum;
                if (r < current_probability) {
                    sets[set_index].Add(pair);
                    break;
                }
                set_index++;
            }
        }

        foreach (var set in sets)
            yield return set;
    }

    /// <summary>
    /// Check if the given file is a binary encoded training set
    /// </summary>
    /// <param name="file">file containing the binary encoded training data</param>
    /// <returns>true if the file is a binary training set</returns>
    public static bool IsBinaryTrainingSet(FileInfo file) {
        using var reader = new BinaryReader(file.OpenRead());
        // Read magic (and validate)
        foreach (var magic in BinaryVectorBuilder.BinaryTrainingSetMagicNumber) {
            if (reader.ReadByte() != magic)
                return false;
        }

        // Read header (and validate)
        var type = (reader.ReadByte());
        var scaling = reader.ReadDouble();
        var output_count = reader.ReadInt32();
        var input_count = reader.ReadInt32();

        if (!Enum.IsDefined(typeof(TrainingVectorStorageType), type))
            return false;
        if (double.IsNaN(scaling) || double.IsInfinity(scaling))
            return false;
        if (output_count < 0)
            return false;
        if (input_count < 0)
            return false;

        return true;
    }

    /// <summary>
    /// Add all vectors stored in binary format to this training data
    /// </summary>
    /// <param name="reader">reader containing binary data</param>
    /// <exception cref="ArgumentException">thrown when vector data-type is unknown</exception>
    public void AddFrom(BinaryReader reader) {
        foreach (var magic in BinaryVectorBuilder.BinaryTrainingSetMagicNumber) {
            if (reader.ReadByte() != magic)
                throw new FormatException("Stream is not formatted as a binary training set");
        }
        var type = (TrainingVectorStorageType)(reader.ReadByte());
        var scaling = reader.ReadDouble();
        var output_count = reader.ReadInt32();
        var input_count = reader.ReadInt32();

        // Outputs
        var outputs = new List<Vec<T>>(output_count);
        for (var i = 0; i < output_count; i++) {
            var vec_size = reader.ReadInt32();
            var data = new T[vec_size];
            for (var j = 0; j < vec_size; j++) {
                var raw = type switch {
                    TrainingVectorStorageType.U8  => (double)reader.ReadByte(),
                    TrainingVectorStorageType.U16 => (double)reader.ReadUInt16(),
                    TrainingVectorStorageType.U32 => (double)reader.ReadUInt32(),
                    TrainingVectorStorageType.U64 => (double)reader.ReadUInt64(),

                    TrainingVectorStorageType.I8  => (double)reader.ReadSByte(),
                    TrainingVectorStorageType.I16 => (double)reader.ReadInt16(),
                    TrainingVectorStorageType.I32 => (double)reader.ReadInt32(),
                    TrainingVectorStorageType.I64 => (double)reader.ReadInt64(),

                    TrainingVectorStorageType.F16 => (double)reader.ReadHalf(),
                    TrainingVectorStorageType.F32 => (double)reader.ReadSingle(),
                    TrainingVectorStorageType.F64 => (double)reader.ReadDouble(),

                    _ => throw new ArgumentException(nameof(TrainingVectorStorageType))
                } * scaling;
                data[j] = (T)Convert.ChangeType(raw, typeof(T));
            }
            outputs.Add( Vec<T>.Wrap(data) );
        }

        for (var i = 0; i < input_count; i++) {
            var output_index = reader.ReadInt32();
            var vec_size = reader.ReadInt32();
            var data = new T[vec_size];
            for (var j = 0; j < vec_size; j++) {
                var raw = type switch {
                    TrainingVectorStorageType.U8  => (double)reader.ReadByte(),
                    TrainingVectorStorageType.U16 => (double)reader.ReadUInt16(),
                    TrainingVectorStorageType.U32 => (double)reader.ReadUInt32(),
                    TrainingVectorStorageType.U64 => (double)reader.ReadUInt64(),

                    TrainingVectorStorageType.I8  => (double)reader.ReadSByte(),
                    TrainingVectorStorageType.I16 => (double)reader.ReadInt16(),
                    TrainingVectorStorageType.I32 => (double)reader.ReadInt32(),
                    TrainingVectorStorageType.I64 => (double)reader.ReadInt64(),

                    TrainingVectorStorageType.F16 => (double)reader.ReadHalf(),
                    TrainingVectorStorageType.F32 => (double)reader.ReadSingle(),
                    TrainingVectorStorageType.F64 => (double)reader.ReadDouble(),

                    _ => throw new ArgumentException(nameof(TrainingVectorStorageType))
                } * scaling;
                data[j] = (T)Convert.ChangeType(raw, typeof(T));
            }
            var input = Vec<T>.Wrap(data);
            var output = outputs[output_index];
            this.Add(input, output);
        }
    }

    /// <summary>
    /// Dump all training data to a binary format
    /// </summary>
    /// <param name="writer">writer to dump vectors to</param>
    public void WriteTo(BinaryWriter writer) {
        // Write magic number
        foreach (var magic in BinaryVectorBuilder.BinaryTrainingSetMagicNumber) {
            writer.Write((byte)magic);
        }

        // Compute number of unique outputs
        var outputs = this.Select(pair => pair.Output).Distinct().ToArray();
        // Compute number of unique inputs
        // Compute vector "scaling" factor
        const double scaling = 1.0; // Assume that scaling was already applied

        // DATA_TYPE SCALING OUT_CLASSES, INPUT_CLASSES
        writer.Write((byte)TrainingVectorStorageType.F64);  // Always write F64
        writer.Write(scaling);                      // Set scaling factor
        writer.Write(outputs.Length);               // Set output count
        writer.Write(this.Size);                    // Set input count

        // Outputs
        foreach (var output in outputs) {
            writer.Write(output.Dimensionality);
            foreach (var element in output) {
                writer.Write((double)Convert.ChangeType(element, typeof(double)));
            }
        }

        // Inputs
        foreach (var pair in this) {
            var output_index = Array.IndexOf(outputs, pair.Output);
            writer.Write(output_index);
            writer.Write(pair.Input.Dimensionality);
            foreach (var element in pair.Input) {
                writer.Write((double)Convert.ChangeType(element, typeof(double)));
            }
        }
    }
}

// Not super useful in this class, but necessary if other utility programs create training data dumps
// eg Images2Dataset using U8 for pixel values
public enum TrainingVectorStorageType : byte {
    U8 = 0b0001_0000,   U16 = 0b0001_0001,  U32 = 0b0001_0010,  U64 = 0b0001_0011,
    I8 = 0b0010_0000,   I16 = 0b0010_0001,  I32 = 0b0010_0010,  I64 = 0b0010_0011,
                        F16 = 0b0100_0001,  F32 = 0b0100_0010,  F64 = 0b0100_0011
}
