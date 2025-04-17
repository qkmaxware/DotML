using System.Linq;
using System.Diagnostics.CodeAnalysis;

namespace DotML.Network.Training;

public class TrainingSetBuilder {
    /// <summary>
    /// Element storage class
    /// </summary>
    public TrainingSet.VectorStorageType StorageType {get; private set;}

    /// <summary>
    /// Scaling factor for all vectors
    /// </summary>
    public double ScalingFactor {get; set;} = 1;

    /// <summary>
    /// Type of value stored in each vector
    /// </summary>
    public Type ValueType => GetValueType(StorageType);

    private static Type GetValueType(TrainingSet.VectorStorageType storageType) {
        switch (storageType) {
            case TrainingSet.VectorStorageType.U8:  return typeof(Byte);
            case TrainingSet.VectorStorageType.U16: return typeof(UInt16);
            case TrainingSet.VectorStorageType.U32: return typeof(UInt32);
            case TrainingSet.VectorStorageType.U64: return typeof(UInt64);

            case TrainingSet.VectorStorageType.I8:  return typeof(SByte);
            case TrainingSet.VectorStorageType.I16: return typeof(Int16);
            case TrainingSet.VectorStorageType.I32: return typeof(Int32);
            case TrainingSet.VectorStorageType.I64: return typeof(Int64);

            case TrainingSet.VectorStorageType.F16: return typeof(Half);
            case TrainingSet.VectorStorageType.F32: return typeof(Single);
            case TrainingSet.VectorStorageType.F64: return typeof(Double);

            default: throw new ArgumentException(nameof(StorageType));
        };
    }

    /// <summary>
    /// Type of array used internally by the builder
    /// </summary>
    public Type ArrayType => GetArrayType(StorageType);

    private static Type GetArrayType(TrainingSet.VectorStorageType storageType) {
        switch (storageType) {
            case TrainingSet.VectorStorageType.U8:  return typeof(Byte[]);
            case TrainingSet.VectorStorageType.U16: return typeof(UInt16[]);
            case TrainingSet.VectorStorageType.U32: return typeof(UInt32[]);
            case TrainingSet.VectorStorageType.U64: return typeof(UInt64[]);

            case TrainingSet.VectorStorageType.I8:  return typeof(SByte[]);
            case TrainingSet.VectorStorageType.I16: return typeof(Int16[]);
            case TrainingSet.VectorStorageType.I32: return typeof(Int32[]);
            case TrainingSet.VectorStorageType.I64: return typeof(Int64[]);

            case TrainingSet.VectorStorageType.F16: return typeof(Half[]);
            case TrainingSet.VectorStorageType.F32: return typeof(Single[]);
            case TrainingSet.VectorStorageType.F64: return typeof(Double[]);

            default: throw new ArgumentException(nameof(StorageType));
        };
    }

    protected TrainingSetBuilder(TrainingSet.VectorStorageType storage) {
        this.StorageType = storage;
        this.pairs = new List<KeyValuePair<Array, Array>>();
    }

    protected TrainingSetBuilder(TrainingSetBuilder builder, bool deep) {
        this.StorageType = builder.StorageType;
        this.ScalingFactor = builder.ScalingFactor;
        if (deep) {
            this.pairs = new List<KeyValuePair<Array, Array>>();
            this.pairs.EnsureCapacity(builder.pairs.Capacity);
            this.pairs.AddRange(builder.pairs);
        } else {
            this.pairs = builder.pairs;
        }
    }

    /// <summary>
    /// Make an array of values that is compatible with the builder's vector elements
    /// </summary>
    /// <param name="size">array size</param>
    /// <returns>array</returns>
    public Array MakeValueArray(int size) {
        return MakeValueArray(this.ArrayType?.GetElementType() ?? typeof(object), size);
    }

    private static Array MakeValueArray(Type valueType, int size) {
        return Array.CreateInstance(valueType, Math.Max(0, size));
    }

    private List<KeyValuePair<Array, Array>> pairs;

    public int InputLength => pairs.Count > 0 ? pairs[0].Key.Length : 0;
    public int OutputLength => pairs.Count > 0 ? pairs[0].Value.Length : 0;

    public IEnumerable<KeyValuePair<Array, Array>> TrainingPairs => pairs.AsReadOnly();

    public KeyValuePair<Array, Array>? GetPair(int index) {
        if (index < 0 || index >= pairs.Count) {
            return null;
        }
        return pairs[index];
    }

    /// <summary>
    /// Number of training pairs
    /// </summary>
    public int Count => pairs.Count;

    /// <summary>
    /// Ensure that the builder has the capacity to store the given number of training pairs
    /// </summary>
    /// <param name="capacity">number of training pairs</param>
    public void EnsureCapacity(int capacity) {
        pairs.EnsureCapacity(capacity);
    }

    /// <summary>
    /// Clear all training pairs
    /// </summary>
    public void Clear() {
        pairs.Clear();
    }

    /// <summary>
    /// Add an input/output vector combination to this builder
    /// </summary>
    /// <param name="input">input vector</param>
    /// <param name="output">output vector</param>
    /// <exception cref="InvalidCastException">thrown if the arrays are not of the appropriate internal type</exception>
    public void Add(Array input, Array output) {
        var type = ArrayType;
        if (input.GetType() != type)
            throw new InvalidCastException(nameof(input));
        if (output.GetType() != type)
            throw new InvalidCastException(nameof(output));

        pairs.Add(new KeyValuePair<Array, Array>(input, output));
    }

    /// <summary>
    /// Insert a training pair at the given index
    /// </summary>
    /// <param name="index">index to insert at</param>
    /// <param name="input">input vector</param>
    /// <param name="output">output vector</param>
    public void Insert(int index, Array input, Array output) {
        pairs.Insert(index, new KeyValuePair<Array, Array>(input, output));
    }

    /// <summary>
    /// Remove the training pair with the given index
    /// </summary>
    /// <param name="index">pair index</param>
    public void RemoveAt(int index) {
        pairs.RemoveAt(index);
    }

    /// <summary>
    /// Remove all training pairs matching the given predicate
    /// </summary>
    /// <param name="predicate">removal condition</param>
    public void RemoveAll(Predicate<KeyValuePair<Array, Array>> predicate) {
        pairs.RemoveAll(predicate);
    }

    private class ArrayComparer : IEqualityComparer<Array> {
        public bool Equals(Array? x, Array? y) {
            if (ReferenceEquals(x, y))
                return true;
            if (x is null || y is null || x.Length != y.Length)
                return false;

            if (x.Length == y.Length)
                return false;            

            var len = x.Length;
            for (var i = 0; i < len; i++) {
                if (x.GetValue(i) != y.GetValue(i))
                    return false;
            }

            return true;
        }

        public int GetHashCode([DisallowNull] Array obj) {
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
    /// Make a training set builder from an existing training set accessed from the given binary reader
    /// </summary>
    /// <param name="reader">reader to read the training set filee</param>
    /// <returns></returns>
    /// <exception cref="FormatException">If the binary reader's stream is not a binary training set</exception>
    /// <exception cref="InvalidCastException">If the storage type is incompatible with this builder implementation</exception>
    public static TrainingSetBuilder From(BinaryReader reader) {
        // Copy of TrainingSet.AddFrom 
        // Read magic number
        var magic = TrainingSet.BinaryTrainingSetMagicNumber;
        for (var i = 0; i < magic.Count; i++) {
            if (reader.ReadByte() != magic[i])
                throw new FormatException("Stream is not formatted as a binary training set");
        }

        var type = (TrainingSet.VectorStorageType)(reader.ReadByte());
        var value_type = GetValueType(type);
        var scaling = reader.ReadDouble();
        var output_count = reader.ReadInt32();
        var input_count = reader.ReadInt32();

        // Outputs
        var outputs = new List<Array>(output_count);
        for (var output_idx = 0; output_idx < output_count; output_idx++) {
            var vec_size = reader.ReadInt32();
            var data = MakeValueArray(value_type, vec_size);
            for (var j = 0; j < vec_size; j++) {
                data.SetValue(type switch {
                    TrainingSet.VectorStorageType.U8  => reader.ReadByte(),
                    TrainingSet.VectorStorageType.U16 => reader.ReadUInt16(),
                    TrainingSet.VectorStorageType.U32 => reader.ReadUInt32(),
                    TrainingSet.VectorStorageType.U64 => reader.ReadUInt64(),

                    TrainingSet.VectorStorageType.I8  => reader.ReadSByte(),
                    TrainingSet.VectorStorageType.I16 => reader.ReadInt16(),
                    TrainingSet.VectorStorageType.I32 => reader.ReadInt32(),
                    TrainingSet.VectorStorageType.I64 => reader.ReadInt64(),

                    TrainingSet.VectorStorageType.F16 => reader.ReadHalf(),
                    TrainingSet.VectorStorageType.F32 => reader.ReadSingle(),
                    TrainingSet.VectorStorageType.F64 => reader.ReadDouble(),

                    _ => throw new InvalidCastException(nameof(TrainingSet.VectorStorageType))
                }, j);
            }
            outputs.Add(data);
        }

        TrainingSetBuilder builder = new TrainingSetBuilder(type);
        builder.ScalingFactor = scaling;
        builder.EnsureCapacity(input_count);

        // Inputs
        for (var input_idx = 0; input_idx < input_count; input_idx++) {
            var output_idx = reader.ReadInt32();
            var vec_size = reader.ReadInt32();
            var data = MakeValueArray(value_type, vec_size);
            for (var j = 0; j < vec_size; j++) {
                data.SetValue(type switch {
                    TrainingSet.VectorStorageType.U8  => reader.ReadByte(),
                    TrainingSet.VectorStorageType.U16 => reader.ReadUInt16(),
                    TrainingSet.VectorStorageType.U32 => reader.ReadUInt32(),
                    TrainingSet.VectorStorageType.U64 => reader.ReadUInt64(),

                    TrainingSet.VectorStorageType.I8  => reader.ReadSByte(),
                    TrainingSet.VectorStorageType.I16 => reader.ReadInt16(),
                    TrainingSet.VectorStorageType.I32 => reader.ReadInt32(),
                    TrainingSet.VectorStorageType.I64 => reader.ReadInt64(),

                    TrainingSet.VectorStorageType.F16 => reader.ReadHalf(),
                    TrainingSet.VectorStorageType.F32 => reader.ReadSingle(),
                    TrainingSet.VectorStorageType.F64 => reader.ReadDouble(),

                    _ => throw new InvalidCastException(nameof(TrainingSet.VectorStorageType))
                }, j);
            }

            var input = data;
            var output = outputs[output_idx];
            builder.Add(input, output);
        }

        return builder;
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
        Array[] outputs = pairs.Select(kv => kv.Value).Distinct(new ArrayComparer()).ToArray();
        // Compute vector "scaling" factor
        double scaling = this.ScalingFactor;

        // DATA_TYPE SCALING OUT_CLASSES, INPUT_CLASSES
        writer.Write((byte)StorageType);            // Storage type is dependent 
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

                    default: throw new InvalidCastException(ValueType.Name);
                };
            }
        }

        // Inputs
        foreach (var pair in pairs) {
            var output_index = Array.IndexOf(outputs, pair.Value);
            writer.Write(output_index);
            writer.Write(pair.Key.Length);
            foreach (var element in pair.Key) {
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

                    default: throw new InvalidCastException(ValueType.Name);
                };
            }
        }
    }

    private static IEnumerable<double> ToIEnumerable(Array array) {
        foreach (var item in array) {
            yield return (double)Convert.ChangeType(item, typeof(Double));
        }
    }

    /// <summary>
    /// Convert the builder's data to a concrete training set
    /// </summary>
    /// <returns>Training set</returns>
    public TrainingSet ToTrainingSet() {
        TrainingSet set = new TrainingSet(this.Count);

        foreach (var pair in TrainingPairs) {
            TrainingPair training = new TrainingPair {
                Input  = Vec<double>.Wrap(ToIEnumerable(pair.Key).Select(v => v * ScalingFactor).ToArray()),
                Output = Vec<double>.Wrap(ToIEnumerable(pair.Value).Select(v => v * ScalingFactor).ToArray()),
            };
            set.Add(training);
        }

        return set;
    }
    
    /// <summary>
    /// Clone this builder to a new instance
    /// </summary>
    /// <returns>TrainingSetBuilder with identical values</returns>
    public TrainingSetBuilder Clone() {
        return new TrainingSetBuilder(this, deep: true);
    }

    /// <summary>
    /// Cast a weakly typed training data builder to a strongly typed one
    /// </summary>
    /// <typeparam name="T">strong value typing</typeparam>
    /// <returns>TrainingSetBuilder for values of type T</returns>
    /// <exception cref="InvalidCastException">thrown if the cast is invalid, like of the wrong type</exception>
    public TrainingSetBuilder<T> Cast<T>() where T:IConvertible {
        var storage_type = typeof(T) switch {
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

            _ => throw new InvalidCastException(nameof(T))
        };
        if (storage_type != this.StorageType) {
            throw new InvalidCastException(nameof(T));
        }

        return new TrainingSetBuilder<T>(this, deep: false);
    }

}

public class TrainingSetBuilder<T> : TrainingSetBuilder where T:IConvertible {

    public TrainingSetBuilder() : base(typeof(T) switch {
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
        }) { }

    public TrainingSetBuilder(TrainingSetBuilder<T> builder) : base(builder, deep: true) { }

    internal TrainingSetBuilder(TrainingSetBuilder builder, bool deep) : base(builder, deep) {
        var determined_type = typeof(T) switch {
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
        if (determined_type != this.StorageType)
            throw new InvalidCastException(nameof(T));
    }

    /// <summary>
    /// Add a training pair
    /// </summary>
    /// <param name="input">input vector values</param>
    /// <param name="output">output vector values</param>
    public void Add(T[] input, T[] output) {
        base.Add(input, output);
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
        foreach (var item in pairs) {
            base.Add(item.Key, item.Value);
        }
    }

    /// <summary>
    /// Add all training pairs
    /// </summary>
    /// <param name="pairs">training pairs</param>
    public void AddRange(IEnumerable<KeyValuePair<T[], T[]>> pairs) {
        foreach (var item in pairs) {
            base.Add(item.Key, item.Value);
        }
    }

    /// <summary>
    /// Add all training pairs
    /// </summary>
    /// <param name="pairs">training pairs</param>
    public void AddRange(IEnumerable<(T[], T[])> pairs) {
        foreach (var item in pairs) {
            base.Add(item.Item1, item.Item2);
        }
    }

    /// <summary>
    /// Remove the given input output pair
    /// </summary>
    /// <param name="input">input vector</param>
    /// <param name="output">output vector</param>
    public void Remove(T[] input, T[] output) {
        RemoveAll((pair) => ReferenceEquals(pair.Key, input) && ReferenceEquals(pair.Value, output));
    }

    /// <summary>
    /// Remove all pairs with the given input vector
    /// </summary>
    /// <param name="input">input vector</param>
    public void RemoveInput(T[] input) {
        RemoveAll((pair) => ReferenceEquals(pair.Key, input));
    }

    /// <summary>
    /// Remove all pairs with the given output vector
    /// </summary>
    /// <param name="output">output vector</param>
    public void RemoveOutput(T[] output) {
        RemoveAll((pair) => ReferenceEquals(pair.Value, output));
    }

    /// <summary>
    /// Insert a training pair at the given index
    /// </summary>
    /// <param name="index">index to insert at</param>
    /// <param name="input">input vector</param>
    /// <param name="output">output vector</param>
    public void Insert(int index, T[] input, T[] output) {
        base.Insert(index, input, output);
    }

}