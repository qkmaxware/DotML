using System.Diagnostics.CodeAnalysis;

namespace DotML.Network.Training;

public class TrainingSetBuilder<T> where T:IConvertible {
    /// <summary>
    /// Storage class
    /// </summary>
    public TrainingSet.VectorStorageType Type {get; private set;}
    
    /// <summary>
    /// Scaling factor for all vectors
    /// </summary>
    public double ScalingFactor {get; private set;} = 1;
    
    public TrainingSetBuilder() {
        this.Type = typeof(T) switch {
            Type u8_type when u8_type == typeof(byte) => TrainingSet.VectorStorageType.U8,   
            Type u16_type when u16_type == typeof(ushort) => TrainingSet.VectorStorageType.U16,  
            Type u32_type when u32_type == typeof(uint) => TrainingSet.VectorStorageType.U32,  
            Type u64_type when u64_type == typeof(ulong) => TrainingSet.VectorStorageType.U64,

            Type i8_type when i8_type == typeof(sbyte) => TrainingSet.VectorStorageType.I8,   
            Type i8_type when i8_type == typeof(short) => TrainingSet.VectorStorageType.I16,  
            Type i8_type when i8_type == typeof(int) => TrainingSet.VectorStorageType.I32,  
            Type i8_type when i8_type == typeof(long) => TrainingSet.VectorStorageType.I64,

            Type f16_type when f16_type == typeof(Half) => TrainingSet.VectorStorageType.F16,  
            Type f32_type when f32_type == typeof(Single) => TrainingSet.VectorStorageType.F32,  
            Type f64_type when f64_type == typeof(Double) => TrainingSet.VectorStorageType.F64,

            _ => throw new ArgumentException(nameof(T))
        };
    }

    private List<(T[] Input, T[] Output)> pairs = new List<(T[] Input, T[] Output)>();

    /// <summary>
    /// Number of training pairs
    /// </summary>
    public int Count => pairs.Count;

    /// <summary>
    /// Clear all training pairs
    /// </summary>
    public void Clear() {
        pairs.Clear();
    }

    /// <summary>
    /// Add a training pair
    /// </summary>
    /// <param name="input">input vector values</param>
    /// <param name="output">output vector values</param>
    public void Add(T[] input, T[] output) {
        pairs.Add((input, output));
    }

    /// <summary>
    /// Add vectors from a generator function
    /// </summary>
    /// <param name="amount">amount to generate</param>
    /// <param name="generator">generator function</param>
    public void AddGenerated(int amount, Func<(T[], T[])> generator) {
        amount = Math.Max(0, amount);
        AddRange(Enumerable.Range(0, amount).Select(i => generator()));
    }

    /// <summary>
    /// Add vectors from a generator function
    /// </summary>
    /// <param name="amount">amount to generate</param>
    /// <param name="generator">generator function</param>
    public void AddGenerated(int amount, Func<KeyValuePair<T[], T[]>> generator) {
        amount = Math.Max(0, amount);
        AddRange(Enumerable.Range(0, amount).Select(i => generator()));
    }

    /// <summary>
    /// Add all training pairs
    /// </summary>
    /// <param name="pairs">training pairs</param>
    public void AddAll(params KeyValuePair<T[], T[]>[] pairs) {
        this.pairs.AddRange(pairs.Select(x => (x.Key, x.Value)));
    }

    /// <summary>
    /// Add all training pairs
    /// </summary>
    /// <param name="pairs">training pairs</param>
    public void AddRange(IEnumerable<KeyValuePair<T[], T[]>> pairs) {
        this.pairs.AddRange(pairs.Select(x => (x.Key, x.Value)));
    }

    /// <summary>
    /// Add all training pairs
    /// </summary>
    /// <param name="pairs">training pairs</param>
    public void AddRange(IEnumerable<(T[], T[])> pairs) {
        this.pairs.AddRange(pairs.Select(x => (x.Item1, x.Item2)));
    }

    /// <summary>
    /// Remove the given input output pair
    /// </summary>
    /// <param name="input">input vector</param>
    /// <param name="output">output vector</param>
    public void Remove(T[] input, T[] output) {
        pairs.RemoveAll((pair) => ReferenceEquals(pair.Input, input) && ReferenceEquals(pair.Output, output));
    }

    /// <summary>
    /// Remove all pairs with the given input vector
    /// </summary>
    /// <param name="input">input vector</param>
    public void RemoveInput(T[] input) {
        pairs.RemoveAll((pair) => ReferenceEquals(pair.Input, input));
    }

    /// <summary>
    /// Remove all pairs with the given output vector
    /// </summary>
    /// <param name="output">output vector</param>
    public void RemoveOutput(T[] output) {
        pairs.RemoveAll((pair) => ReferenceEquals(pair.Output, output));
    }

    /// <summary>
    /// Remove the training pair with the given index
    /// </summary>
    /// <param name="index">pair index</param>
    public void RemoveAt(int index) {
        pairs.RemoveAt(index);
    }

    /// <summary>
    /// Insert a training pair at the given index
    /// </summary>
    /// <param name="index">index to insert at</param>
    /// <param name="input">input vector</param>
    /// <param name="output">output vector</param>
    public void Insert(int index, T[] input, T[] output) {
        pairs.Insert(index, (input, output));
    }

    private class ArrayComparer : IEqualityComparer<T[]> {
        public bool Equals(T[]? x, T[]? y) {
            if (ReferenceEquals(x, y))
                return true;
            if (x is null || y is null || x.Length != y.Length)
                return false;

            return x.Length == y.Length && x.SequenceEqual(y);
        }

        public int GetHashCode([DisallowNull] T[] obj) {
            if (obj is null)
                return 0;

            unchecked {
                int hash = 17;
                foreach (var item in obj) {
                    hash = hash * 31 + (item.GetHashCode());
                }
                return hash;
            }
        }
    }

    /// <summary>
    /// Dump all training data to a binary format
    /// </summary>
    /// <param name="writer">writer to dump vectors to</param>
    public void WriteTo(BinaryWriter writer) {
        // Write magic number
        var magic = TrainingSet.BinaryTrainingSetMagicNumber;
        for (var i = 0; i < magic.Count; i++) {
            writer.Write((byte)magic[i]);
        }

        // Compute number of unique outputs
        var outputs = pairs.Select(pair => pair.Output).Distinct(new ArrayComparer()).ToArray();
        // Compute vector "scaling" factor
        double scaling = this.ScalingFactor;

        // DATA_TYPE SCALING OUT_CLASSES, INPUT_CLASSES
        writer.Write((byte)Type);                   // Storage type is dependent 
        writer.Write(scaling);                      // Set scaling factor
        writer.Write(outputs.Length);               // Set output count
        writer.Write(pairs.Count);                  // Set input count

        // Outputs
        foreach (var output in outputs) {
            writer.Write(output.Length);
            foreach (var element in output) {
                switch (element) {
                    case byte u8_value:  writer.Write(u8_value); break;
                    case ushort u16_type: writer.Write(u16_type); break;
                    case uint u32_type: writer.Write(u32_type); break;
                    case ulong u64_type: writer.Write(u64_type); break;

                    case sbyte i8_type:   writer.Write(i8_type); break;
                    case short i16_type:  writer.Write(i16_type); break;
                    case int i32_type:   writer.Write(i32_type); break;
                    case long i64_type: writer.Write(i64_type); break;

                    case Half f16_type: writer.Write(f16_type); break;
                    case Single f32_type:  writer.Write(f32_type); break;
                    case Double f64_type: writer.Write(f64_type); break;

                    default: throw new ArgumentException(nameof(T));
                };
            }
        }

        // Inputs
        foreach (var pair in pairs) {
            var output_index = Array.IndexOf(outputs, pair.Output);
            writer.Write(output_index);
            writer.Write(pair.Input.Length);
            foreach (var element in pair.Input) {
                switch (element) {
                    case byte u8_value:  writer.Write(u8_value); break;
                    case ushort u16_type: writer.Write(u16_type); break;
                    case uint u32_type: writer.Write(u32_type); break;
                    case ulong u64_type: writer.Write(u64_type); break;

                    case sbyte i8_type:   writer.Write(i8_type); break;
                    case short i16_type:  writer.Write(i16_type); break;
                    case int i32_type:   writer.Write(i32_type); break;
                    case long i64_type: writer.Write(i64_type); break;

                    case Half f16_type: writer.Write(f16_type); break;
                    case Single f32_type:  writer.Write(f32_type); break;
                    case Double f64_type: writer.Write(f64_type); break;

                    default: throw new ArgumentException(nameof(T));
                };
            }
        }
    }

    /// <summary>
    /// Convert the builder's data to a concrete training set
    /// </summary>
    /// <returns>Training set</returns>
    public TrainingSet ToTrainingSet() {
        TrainingSet set = new TrainingSet(this.pairs.Count);

        foreach (var pair in pairs) {
            TrainingPair training = new TrainingPair {
                Input  = Vec<double>.Wrap(pair.Input.Select(v => v.ToDouble(null) * ScalingFactor).ToArray()),
                Output = Vec<double>.Wrap(pair.Output.Select(v => v.ToDouble(null) * ScalingFactor).ToArray()),
            };
            set.Add(training);
        }

        return set;
    }
}