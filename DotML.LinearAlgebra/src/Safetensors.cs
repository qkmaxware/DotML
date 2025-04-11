using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace DotML;

/// <summary>
/// Class to aid in the creation of safetensors files
/// </summary>
public class Safetensors {
    // Generic tensor storage
    private class UnknownObjectTensor : ITensorLike<object?> {
        public GenericShape shape;
        public object data;

        public UnknownObjectTensor(GenericShape shape, object data) {
            this.shape = shape;
            this.data = data;
        }

        public int Dimensions => shape.Dimensions;

        public int GetDimension(int index) => shape.GetDimension(index);

        public object? GetElementAt(params int[] indices) {
            var index = create_1d_index(shape.Lengths, indices);
            return ((Array)data).GetValue(index);
        }
    }

    // Generic tensor shape
    private struct GenericShape : IShape {
        private int[] shape;

        public int Size => shape.Aggregate(1, (lhs, rhs) => lhs * rhs);

        public int[] Lengths => shape;

        public GenericShape(int[] shape) {
            this.shape = shape;
        }

        public int Dimensions => shape.Length;

        public int GetDimension(int index) {
            if (index >= 0 && index < shape.Length)
                return shape[index];
            return 1;
        }
    }
    private Dictionary<string, UnknownObjectTensor> tensors = new Dictionary<string, UnknownObjectTensor>();

    /// <summary>
    /// All keys in this safetensors set
    /// </summary>
    /// <returns>enumerable of keys</returns>
    public IEnumerable<string> Keys() => tensors.Keys;

    /// <summary>
    /// Get the shape of the given tensor
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public IShape ShapeOf(string key) {
        if (tensors.TryGetValue(key, out var tensor)) {
            return tensor.shape;
        }
        return new Shape2D();
    }

    /// <summary>
    /// Get the type of value used for each element in the given tensor
    /// </summary>
    /// <param name="key">tensor name</param>
    /// <returns>element type</returns>
    public Type TypeOf(string key) {
        if (tensors.TryGetValue(key, out var tensor)) {
            return tensor.data?.GetType()?.GetElementType() ?? typeof(object);
        }

        return typeof(object);
    }

    /// <summary>
    /// Check if this safetensors set contains a tensor with the given name
    /// </summary>
    /// <param name="key">key name</param>
    /// <returns>true if key exists</returns>
    public bool ContainsKey(string key) => tensors.ContainsKey(key);

    /// <summary>
    /// Rename a key from one name to another. The old key must exist and the new key must not already exist.
    /// </summary>
    /// <param name="fromKey">The key to rename from, must be present already</param>
    /// <param name="toKey">The key to rename to, must not already exist</param>
    /// <returns>true if renaming was successful</returns>
    public bool RenameKey(string fromKey, string toKey) {
        // Check that the fromKey exists and that the toKey does not
        if (!tensors.ContainsKey(fromKey)) {
            return false;
        }
        if (tensors.ContainsKey(toKey)) {
            return false;
        }

        // Do the renaming (remove value with old key, add same value with new key)
        var tensor = tensors[fromKey];
        tensors.Remove(fromKey);
        tensors.Add(toKey, tensor);
        return true;
    }

    /// <summary>
    /// Get the tensor associated with the given key
    /// </summary>
    /// <typeparam name="TOut">output type</typeparam>
    /// <param name="key">tensor key</param>
    /// <returns>matrix</returns>
    /// <exception cref="KeyNotFoundException">thrown when the given key doesn't exist in the safetensors set</exception>
    public Matrix<TOut> GetTensor<TOut>(string key) where TOut:INumber<TOut> { // TODO rename this to GetMatrix or GetTensorAsMatrix
        if (!tensors.TryGetValue(key, out var tensor)) {
            throw new KeyNotFoundException(key);
        }
        var shape = tensor.shape;
        if (shape.Dimensions != 2)
            throw new ArgumentException($"Cannot load a tensor of dimensionality {shape.Dimensions} into a 2d matrix");

        if (tensor.data is TOut[] elements && Matrix<TOut>.IsRowMajor) {
            // No additional memory allocation, just use the array as is. 
            return Matrix<TOut>.FromFlattened(shape.GetDimension(0), shape.GetDimension(1), elements);
        } else {
            var new_matrix = new Matrix<TOut>(shape.GetDimension(0), shape.GetDimension(1));
            LoadTensorInto<Matrix<TOut>, TOut>(key, new_matrix);
            return new_matrix;
        }
    }

    /// <summary>
    /// Get the tensor associated with the given key
    /// </summary>
    /// <typeparam name="TOut">output type</typeparam>
    /// <param name="key">tensor key</param>
    /// <returns>matrix</returns>
    /// <exception cref="KeyNotFoundException">thrown when the given key doesn't exist in the safetensors set</exception>
    public GenericTensor<TElement> GetTensorData<TElement>(string key) {
        if (!tensors.TryGetValue(key, out var tensor)) {
            throw new KeyNotFoundException(key);
        }
        var shape = tensor.shape;
        var obj = new GenericTensor<TElement>(shape.Lengths);
        LoadTensorInto<GenericTensor<TElement>, TElement>(key, obj);
        return obj;
    }

    // Desired usage Matrix<double> matrix = GetTensorAs<Matrix<double>, double>(key);
    internal static IEnumerable<int[]> iterate_over_dimensions(int[] shape) {
        int[] indices = new int[shape.Length];
        while (true) {
            // Return the combination
            yield return indices;

            // Find the rightmost dimension to increment
            int i = shape.Length - 1;
            while (i >= 0) {
                indices[i]++;
                if (indices[i] < shape[i]) // If the index is within bounds
                {
                    break;
                }
                indices[i] = 0; // Reset to 0 if out of bounds and continue incrementing left
                i--;
            }

            // If all indices have been exhausted (i.e., we've generated all combinations), break the loop
            if (i < 0)
                break;
        }
    }
    internal static int create_1d_index(int[] shape, int[] indices) {
        int index = 0;
        int stride = 1;

        // Loop over the dimensions in reverse order to compute the index
        for (int i = shape.Length - 1; i >= 0; i--) {
            index += indices[i] * stride;
            stride *= shape[i]; // Update the stride for the next dimension
        }

        return index;
    }
    
    /// <summary>
    /// Load the given tensor values into the result tensor object
    /// </summary>
    /// <typeparam name="TTensor">type of tensor object</typeparam>
    /// <typeparam name="TElement">type of tensor elements</typeparam>
    /// <param name="key">tensor name</param>
    /// <param name="result">tensor result object</param>
    /// <exception cref="KeyNotFoundException">thrown when no tensor exists with the provided key</exception>
    /// <exception cref="IndexOutOfRangeException">thrown when the shape of the tensor doesn't match the shape of the result object</exception>
    public void LoadTensorInto<TTensor, TElement>(string key, TTensor result) where TTensor:IMutableTensorLike<TElement> {
        if (!tensors.TryGetValue(key, out var tensor)) {
            throw new KeyNotFoundException(key);
        }
        var shape = Enumerable.Range(0, result.Dimensions).Select(x => result.GetDimension(x)).ToArray();
        if (!tensor.shape.Lengths.SequenceEqual(shape)) {
            throw new IndexOutOfRangeException($"Resulting shape {string.Join('x', shape)} doesn't match shape of tensor {key} {string.Join('x', tensor.shape.Lengths)}.");
        }

        var data = (Array)tensor.data;
        foreach (var indices in iterate_over_dimensions(shape)) {
            var index1d = create_1d_index(shape, indices);
            var value = (TElement?)Convert.ChangeType(data.GetValue(index1d), typeof(TElement));
            if (value is not null)
                result.SetElementAt(value, indices);
        }
    }

    /// <summary>
    /// Add any tensor-like object to the safetensor
    /// </summary>
    /// <typeparam name="T">stored element type</typeparam>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add<T>(string name, ITensorLike<T> tensor) {
        // Create a shape that matches the generic tensor
        var shape = new int[tensor.Dimensions];
        for (var i = 0; i < shape.Length; i++)
            shape[i] = tensor.GetDimension(i);
        var count = shape.Aggregate(1, (lhs, rhs) => lhs * rhs);

        // Copy the tensor elements into a row-major order 1d slice
        T[] values = new T[count];
        foreach (var indices in iterate_over_dimensions(shape)) {
            var index1d = create_1d_index(shape, indices);
            values[index1d] = tensor.GetElementAt(indices);
        }

        // Create a generic tensor to store with the provided shape and data
        UnknownObjectTensor to_store = new UnknownObjectTensor(
            new GenericShape(shape),
            values
        );

        // Store it
        this.tensors.Add(name, to_store);
    }

    /// <summary>
    /// Add a 16bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add(string name, Matrix<Half> matrix) { 
        this.tensors.Add(name, new UnknownObjectTensor (shape: new GenericShape(new int[]{ matrix.Rows, matrix.Columns }), data: matrix.AsRowMajorArray() ));
    }
    /// <summary>
    /// Add a 16bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add(string name, Vec<Half> vector) { 
        this.tensors.Add(name, new UnknownObjectTensor (shape: new GenericShape(new int[]{ vector.Dimensionality, 1 }), data: vector.AsArray() ));
    }
    /// <summary>
    /// Add a 32bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add(string name, Matrix<float> matrix) { 
        this.tensors.Add(name, new UnknownObjectTensor (shape: new GenericShape(new int[]{ matrix.Rows, matrix.Columns }), data: matrix.AsRowMajorArray()  ));
    }
    /// <summary>
    /// Add a 32bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add(string name, Vec<float> vector) {
        this.tensors.Add(name, new UnknownObjectTensor (shape: new GenericShape(new int[]{ vector.Dimensionality, 1 }), data: vector.AsArray() ));
    }
    /// <summary>
    /// Add a 64bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add(string name, Matrix<double> matrix) {
        this.tensors.Add(name, new UnknownObjectTensor (shape: new GenericShape(new int[]{ matrix.Rows, matrix.Columns }), data: matrix.AsRowMajorArray() ));
    }
    /// <summary>
    /// Add a 64bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add(string name, Vec<double> vector) { 
        this.tensors.Add(name, new UnknownObjectTensor (shape: new GenericShape(new int[]{ vector.Dimensionality, 1 }), data: vector.AsArray() ));
    }

    /// <summary>
    /// Read the tensors from the give file
    /// </summary>
    /// <param name="file">file pointer</param>
    public static Safetensors ReadFromFile(FileInfo file) {
        using var writer = new BinaryReader(file.OpenRead());
        return ReadFrom(writer);
    }

    /// <summary>
    /// Read the tensors from the give file
    /// </summary>
    /// <param name="path">file path</param>
    public static Safetensors ReadFromFile(string path) {
        using var writer = new BinaryReader(File.Open(path, FileMode.Open));
        return ReadFrom(writer);
    }

    /// <summary>
    /// Load safetensors from a binary reader
    /// </summary>
    /// <param name="reader">binary reader of a safetensors set</param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException">thrown when a feature is not supported in this library</exception>
    /// <exception cref="FormatException">thrown when the safetensors format is not followed</exception>
    /// <exception cref="NullReferenceException">thrown when necessary information is missing</exception>
    public static Safetensors ReadFrom(BinaryReader reader) {
        ulong base_stream_pos = 0U; // Manually track the stream position. Every time we read we need to increment appropriately.
        var header_size = reader.ReadUInt64();
        var header_string = System.Text.Encoding.UTF8.GetString(reader.ReadBytes((int)header_size));
        var header_json = JsonSerializer.Deserialize<Dictionary<string, TensorInfo>>(header_string);
        base_stream_pos += header_size + sizeof(UInt64); // header_size (bytes) + size_of_uint64 (bytes)
        
        Safetensors sb = new Safetensors();
        var buffer_offset = base_stream_pos;
        if (header_json is not null) {
            foreach (var entry in header_json.OrderBy(x => x.Value.DataStartOffset())) {
                // Read header
                var key         = entry.Key;
                var tensorInfo  = entry.Value;
                var shape       = new GenericShape(tensorInfo.shape ?? []);

                var dimensions  = shape.Dimensions;
                var entries     = shape.Size;
                var startIndex  = (ulong)tensorInfo.DataStartOffset();
                var endIndex    = (ulong)tensorInfo.DataEndOffset();


                if (endIndex < startIndex) {
                    throw new FormatException($"Tensor {key} has invalid data offsets [{startIndex}, {endIndex}].");
                }

                // Allocate storage
                var data        = make_array(tensorInfo.dtype, entries);
                var type        = data.GetType().GetElementType() ?? typeof(object);
                var matrix_byte_size = (ulong)(Marshal.SizeOf(type) * shape.Size);
                if (endIndex != (startIndex + matrix_byte_size)) {
                    throw new FormatException($"Tensor {key} data offset length doesn't match element count and sizeof data type.");
                }

                // Fast-forward (just read until the start position)
                var start_position = buffer_offset + startIndex;
                var end_position = buffer_offset + endIndex;
                if (base_stream_pos > start_position) {
                    throw new FormatException($"Tensor {key} data offset overlaps with another tensor.");
                }
                while (base_stream_pos < start_position) {
                    reader.ReadByte();
                    base_stream_pos++;
                }

                // Read values into storage
                for (var i = 0; i < entries; i++) {
                    switch (data) {
                        case SByte[] i8_data: i8_data[i]    = reader.ReadSByte(); base_stream_pos += sizeof(SByte); break;
                        case Int16[] i16_data: i16_data[i]  = reader.ReadInt16(); base_stream_pos += sizeof(Int16); break;
                        case Int32[] i32_data: i32_data[i]  = reader.ReadInt32(); base_stream_pos += sizeof(Int32); break;
                        case Int64[] i64_data: i64_data[i]  = reader.ReadInt64(); base_stream_pos += sizeof(Int64); break;

                        case Byte[] u8_data: u8_data[i]     = reader.ReadByte();   base_stream_pos += sizeof(Byte);   break;
                        case UInt16[] u16_data: u16_data[i] = reader.ReadUInt16(); base_stream_pos += sizeof(UInt16); break;
                        case UInt32[] u32_data: u32_data[i] = reader.ReadUInt32(); base_stream_pos += sizeof(UInt32); break;
                        case UInt64[] u64_data: u64_data[i] = reader.ReadUInt64(); base_stream_pos += sizeof(UInt64); break;

                        case Half[] f16_data: f16_data[i]   = reader.ReadHalf();   base_stream_pos += sizeof(UInt16); break;
                        case Single[] f32_data: f32_data[i] = reader.ReadSingle(); base_stream_pos += sizeof(Single); break;
                        case Double[] f64_data: f64_data[i] = reader.ReadDouble(); base_stream_pos += sizeof(Double); break;

                        default:
                            throw new ArgumentException($"Type {data.GetType()} is not supported");
                    }
                }
                sb.tensors.Add(
                    key, 
                    new UnknownObjectTensor(shape, data)
                );
            }
        }

        return sb;
    }

    /// <summary>
    /// Write the tensors to the give file
    /// </summary>
    /// <param name="file">file pointer</param>
    public void WriteToFile(FileInfo file) {
        using var writer = new BinaryWriter(file.OpenWrite());
        WriteTo(writer);
    }

    /// <summary>
    /// Write the tensors to the give file
    /// </summary>
    /// <param name="path">file path</param>
    public void WriteToFile(string path) {
        using var writer = new BinaryWriter(File.Open(path, FileMode.Create));
        WriteTo(writer);
    }

    private class TensorInfo {
        public string? dtype {get; set;}
        public int[]? shape {get; set;}
        public ulong[]? data_offsets {get; set;}
        public ulong DataStartOffset() => data_offsets?.ElementAtOrDefault(0) ?? 0;
        public ulong DataEndOffset() => data_offsets?.ElementAtOrDefault(1) ?? 0;
        public Dictionary<string, string>? __metadata__ {get; set;}
    }

    private static object make_array(string? t, int count) {
        t = t?.ToUpper();

        if        (t == "I8") {
            return new SByte[count];
        } else if (t == "I16") {
            return new Int16[count];
        } else if (t == "I32") {
            return new Int32[count];
        } else if (t == "I64") {
            return new Int64[count];
        } else if (t == "I128") {
            return new Int128[count];
        } else if (t == "U8") {
            return new Byte[count];
        } else if (t == "U16") {
            return new UInt16[count];
        } else if (t == "U32") {
            return new UInt32[count];
        } else if (t == "U64") {
            return new UInt64[count];
        } else if (t == "U128") {
            return new UInt128[count];
        } else if (t == "F16") {
            return new Half[count];
        } else if (t == "F32") {
            return new Single[count];
        } else if (t == "F64") {
            return new Double[count];
        } 

        else {
            throw new ArgumentException($"Type {t} is not supported");
        }
    }

    private static string type_to_string(Type t) {
        if        (t == typeof(SByte)) {
            return "I8";
        } else if (t == typeof(Int16)) {
            return "I16";
        } else if (t == typeof(Int32)) {
            return "I32";
        } else if (t == typeof(Int64)) {
            return "I64";
        } else if (t == typeof(Int128)) {
            return "I128";
        } else if (t == typeof(Byte)) {
            return "U8";
        } else if (t == typeof(UInt16)) {
            return "U16";
        } else if (t == typeof(UInt32)) {
            return "U32";
        } else if (t == typeof(UInt64)) {
            return "U64";
        } else if (t == typeof(UInt128)) {
            return "U128";
        } else if (t == typeof(Half)) {
            return "F16";
        } else if (t == typeof(Single)) {
            return "F32";
        } else if (t == typeof(Double)) {
            return "F64";
        } 

        else {
            throw new ArgumentException($"Type {t} is not supported");
        }
    }

    private static void create_header_for(Dictionary<string, TensorInfo> header, ref ulong buffer_offset, Dictionary<string, UnknownObjectTensor> tensors) {
        foreach (var matrix in tensors) {
            var shape = matrix.Value.shape;
            var type = matrix.Value.data?.GetType()?.GetElementType() ?? typeof(object);
            var matrix_byte_size = (ulong)(Marshal.SizeOf(type) * shape.Size);
            header.Add(
                matrix.Key,
                new TensorInfo { 
                    dtype           = type_to_string(type),
                    shape           = shape.Lengths,
                    data_offsets    = new ulong[]{ buffer_offset, buffer_offset + matrix_byte_size },
                    __metadata__    = new Dictionary<string, string>{
                        // TODO 
                    }
                }
            );
            buffer_offset = buffer_offset + matrix_byte_size;
        }
    }

    /// <summary>
    /// Write the tensors to the given binary writer
    /// </summary>
    /// <param name="writer">writer</param>
    public void WriteTo(BinaryWriter writer) {
        // Header
        var header = new Dictionary<string, TensorInfo>();
        var buffer_offset = 0ul;
        create_header_for(header, ref buffer_offset, tensors);
        var json = JsonSerializer.Serialize(header);
        var header_bytes = System.Text.Encoding.UTF8.GetBytes(json);
        writer.Write(header_bytes.LongLength);
        writer.Write(header_bytes);

        // Tensors
        foreach (var tensor_pair in tensors) {
            var tensor = tensor_pair.Value;
            switch (tensor.data) {
                case SByte[] i8_data: foreach (var v in i8_data) { writer.Write(v); } break;
                case Int16[] i16_data: foreach (var v in i16_data) { writer.Write(v); } break;
                case Int32[] i32_data: foreach (var v in i32_data) { writer.Write(v); } break;
                case Int64[] i64_data: foreach (var v in i64_data) { writer.Write(v); } break;

                case Byte[] u8_data: foreach (var v in u8_data) { writer.Write(v); } break;
                case UInt16[] u16_data: foreach (var v in u16_data) { writer.Write(v); } break;
                case UInt32[] u32_data: foreach (var v in u32_data) { writer.Write(v); } break;
                case UInt64[] u64_data: foreach (var v in u64_data) { writer.Write(v); } break;

                case Half[] f16_data: foreach (var v in f16_data) { writer.Write(v); } break;
                case Single[] f32_data: foreach (var v in f32_data) { writer.Write(v); } break;
                case Double[] f64_data: foreach (var v in f64_data) { writer.Write(v); } break;

                default:
                    throw new ArgumentException($"Type {tensor.data.GetType()} is not supported");
            }
        }
    }
}