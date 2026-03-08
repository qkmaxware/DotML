using System.ComponentModel;
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
        public Shape shape;
        public object data;

        public UnknownObjectTensor(Shape shape, object data) {
            this.shape = shape;
            this.data = data;
        }

        public int Len() => ((Array)data).Length;

        public int Rank => shape.Rank;

        public int GetDimension(int index) => shape.Length(index);

        public object? GetElementAt(params int[] indices) {
            var index = create_1d_index(shape.AsDimensionSpan(), indices);
            return ((Array)data).GetValue(index);
        }
    }

    private Dictionary<string, UnknownObjectTensor> tensors = new Dictionary<string, UnknownObjectTensor>();
    private Dictionary<string, Dictionary<string, string>> tensor_metadata = new Dictionary<string, Dictionary<string, string>>();

    /// <summary>
    /// Number of saved tensors
    /// </summary>
    public int Count => tensors.Count;

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
    /// Test a tensor in the safetensor file for to see if nay of its elements match the given predicate condition
    /// </summary>
    /// <param name="key">tensor key</param>
    /// <param name="predicate">condition to run on each element</param>
    /// <returns>true if the tensor exists and any element matches the condition</returns>
    public bool AnyIn(string key, Func<object?, bool> predicate)
    {
        if (tensors.TryGetValue(key, out var tensor)) {
            Array arr = (Array)tensor.data;
            for (var i = 0; i < arr.Length; i++)
            {
                if (predicate(arr.GetValue(i)))
                    return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Gets the metadata associated with the given tensor
    /// </summary>
    /// <param name="key">Tensor metadata</param>
    /// <returns>metadata collection</returns>
    /// <exception cref="KeyNotFoundException">Thrown if the tensor with the given key doesn't exist</exception>
    public Dictionary<string, string> MetadataOf(string key) {
        if (tensor_metadata.TryGetValue(key, out var metadata)) {
            return metadata;
        }
        if (!ContainsKey(key)) {
            throw new KeyNotFoundException($"Tensor {key} not found, have you added it?");
        }

        var dict = new Dictionary<string, string>();
        tensor_metadata.Add(key, dict);
        return dict;
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
        if (tensor_metadata.TryGetValue(fromKey, out var metadata)) {
            tensor_metadata.Remove(fromKey);
            tensor_metadata.Add(toKey, metadata);
        }
        
        return true;
    }

    /// <summary>
    /// Get the tensor associated with the given key (tensor must be rank 1)
    /// </summary>
    /// <typeparam name="TOut">output type</typeparam>
    /// <param name="key">tensor key</param>
    /// <returns>vector</returns>
    /// <exception cref="KeyNotFoundException">thrown when the given key doesn't exist in the safetensors set</exception>
    public Vec<TOut> GetVector<TOut>(string key) where TOut:INumber<TOut> { // TODO rename this to GetMatrix or GetTensorAsMatrix
        if (!tensors.TryGetValue(key, out var tensor)) {
            throw new KeyNotFoundException(key);
        }
        var shape = tensor.shape;
        if (shape.Rank != 1)
            throw new ArgumentException($"Cannot load a tensor of dimensionality {shape.Rank} into a vector");

        if (tensor.data is TOut[] elements) {
            // No additional memory allocation, just use the array as is. 
            return new Vec<TOut>(elements);
        } else {
            var new_matrix = new Vec<TOut>(tensor.Len());
            LoadTensorInto<Vec<TOut>, TOut>(key, new_matrix);
            return new_matrix;
        }
    }

    /// <summary>
    /// Get the tensor associated with the given key (tensor must be rank 2)
    /// </summary>
    /// <typeparam name="TOut">output type</typeparam>
    /// <param name="key">tensor key</param>
    /// <returns>matrix</returns>
    /// <exception cref="KeyNotFoundException">thrown when the given key doesn't exist in the safetensors set</exception>
    public Matrix<TOut> GetMatrix<TOut>(string key) where TOut:INumber<TOut> { // TODO rename this to GetMatrix or GetTensorAsMatrix
        if (!tensors.TryGetValue(key, out var tensor)) {
            throw new KeyNotFoundException(key);
        }
        var shape = tensor.shape;
        if (shape.Rank != 2)
            throw new ArgumentException($"Cannot load a tensor of dimensionality {shape.Rank} into a 2d matrix");

        if (tensor.data is TOut[] elements && Matrix<TOut>.IsRowMajor) {
            // No additional memory allocation, just use the array as is. 
            return Matrix<TOut>.FromFlattened(shape.Length(0), shape.Length(1), elements);
        } else {
            var new_matrix = new Matrix<TOut>(shape.Length(0), shape.Length(1));
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
    public Tensor<TElement> GetTensor<TElement>(string key) where TElement : INumber<TElement> {
        if (!tensors.TryGetValue(key, out var tensor)) {
            throw new KeyNotFoundException(key);
        }
        var shape = tensor.shape;
        if (tensor.data is TElement[] elements) {
            // No additional memory allocation, just use the array as is. 
            return Tensor<TElement>.FromFlattenedArray(shape, elements);
        } else {
            // Allocate a new array transform the data and copy it
            var obj = Tensor<TElement>.Defaults(shape);
            // Since this is all stored in row-major order there is a faster way to do this
            //var converter = TypeDescriptor.GetConverter(typeof(TElement));
            TElement[] results = obj.AsArray();
            for (var i = 0; i < results.Length; i++) {
                //results[i] = (TElement)converter.ConvertFrom(((Array)tensor.data).GetValue(i));
                results[i] = (TElement)Convert.ChangeType(((Array)tensor.data).GetValue(i), typeof(TElement));
            }
            //LoadTensorInto<GenericTensor<TElement>, TElement>(key, obj);
            return obj;
        }
    }

    /// <summary>
    /// Get the tensor data associated with the given key
    /// </summary>
    /// <typeparam name="TOut">output type</typeparam>
    /// <param name="key">tensor key</param>
    /// <returns>matrix</returns>
    /// <exception cref="KeyNotFoundException">thrown when the given key doesn't exist in the safetensors set</exception>
    public GenericTensor<TElement> GetData<TElement>(string key) {
        if (!tensors.TryGetValue(key, out var tensor)) {
            throw new KeyNotFoundException(key);
        }
        var shape = tensor.shape;
        if (tensor.data is TElement[] elements) {
            // No additional memory allocation, just use the array as is. 
            return new GenericTensor<TElement>(shape.ToArray(), elements);
        } else {
            // Allocate a new array transform the data and copy it
            TElement[] results = new TElement[shape.LogicalElementCount()];
            var obj = new GenericTensor<TElement>(shape.ToArray(), results);
            // Since this is all stored in row-major order there is a faster way to do this
            //var converter = TypeDescriptor.GetConverter(typeof(TElement));
            for (var i = 0; i < results.Length; i++) {
                //results[i] = (TElement)converter.ConvertFrom(((Array)tensor.data).GetValue(i));
                results[i] = (TElement)Convert.ChangeType(((Array)tensor.data).GetValue(i), typeof(TElement));
            }
            //LoadTensorInto<GenericTensor<TElement>, TElement>(key, obj);
            return obj;
        }
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
    internal static int create_1d_index(ReadOnlySpan<int> shape, int[] indices) {
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
        var shape = Enumerable.Range(0, result.Rank).Select(x => result.GetDimension(x)).ToArray();
        if (!tensor.shape.AsDimensionEnumerable().SequenceEqual(shape)) {
            throw new IndexOutOfRangeException($"Resulting shape {string.Join('x', shape)} doesn't match shape of tensor {key} {string.Join('x', tensor.shape.AsDimensionEnumerable())}.");
        }

        //var converter = TypeDescriptor.GetConverter(typeof(TElement));
        var data = (Array)tensor.data;
        foreach (var indices in iterate_over_dimensions(shape)) {
            var index1d = create_1d_index(shape, indices);
            //var value = (TElement?)converter.ConvertFrom(data.GetValue(index1d));
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
        var shape = new int[tensor.Rank];
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
            new Shape(shape),
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
        this.tensors.Add(name, new UnknownObjectTensor (shape: new Shape(new int[]{ matrix.Rows, matrix.Columns }), data: matrix.AsRowMajorArray() ));
    }
    
    /// <summary>
    /// Add a 16bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add(string name, Vec<Half> vector) { 
        this.tensors.Add(name, new UnknownObjectTensor (shape: new Shape(new int[]{ vector.Dimensionality, 1 }), data: vector.AsArray() ));
    }

    /// <summary>
    /// Add a 32bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add(string name, Matrix<float> matrix) { 
        this.tensors.Add(name, new UnknownObjectTensor (shape: new Shape(new int[]{ matrix.Rows, matrix.Columns }), data: matrix.AsRowMajorArray()  ));
    }

    /// <summary>
    /// Add a 32bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add(string name, Vec<float> vector) {
        this.tensors.Add(name, new UnknownObjectTensor (shape: new Shape(new int[]{ vector.Dimensionality }), data: vector.AsArray() ));
    }

    /// <summary>
    /// Add a 64bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add(string name, Matrix<double> matrix) {
        this.tensors.Add(name, new UnknownObjectTensor (shape: new Shape(new int[]{ matrix.Rows, matrix.Columns }), data: matrix.AsRowMajorArray() ));
    }

    /// <summary>
    /// Add a 64bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor</param>
    public void Add(string name, Vec<double> vector) { 
        this.tensors.Add(name, new UnknownObjectTensor (shape: new Shape(new int[]{ vector.Dimensionality, 1 }), data: vector.AsArray() ));
    }

    public static readonly string QuantizationMethodKey = "quantization.method";
    public static readonly string QuantizationScaleKey = "quantization.scale";
    public static readonly string QuantizationZeroPointKey = "quantization.zero-point";

    /// <summary>
    /// Quantize all compatible tensors in the safetensors set using the given quantizer.
    /// Only tensors of the type TIn will be quantized to the type TOut and the others will be skipped.
    /// </summary>
    /// <typeparam name="TIn">Tensor input type</typeparam>
    /// <typeparam name="TOut">Quantized tensor output type</typeparam>
    /// <param name="quantizer">Quantization method</param>
    public void QuantizeAll<TIn, TOut>(IQuantization<TIn, TOut> quantizer) {
        foreach (var key in this.Keys()) {
            Quantize(key, quantizer);
        }
    }

    /// <summary>
    /// Quantize the given tensors in the safetensors set using the given quantizer.
    /// Only tensors of the type TIn will be quantized to the type TOut if the tensor is not of type TIn this method does nothing.
    /// </summary>
    /// <typeparam name="TIn">Tensor input type</typeparam>
    /// <typeparam name="TOut">Quantized tensor output type</typeparam>
    /// <param name="quantizer">Quantization method</param>
    public void Quantize<TIn, TOut>(string key, IQuantization<TIn, TOut> quantizer) {
        var tensor = this.tensors[key];
        if (tensor.data is not TIn[] data) {
            // This quantizer only works for tensors of the type TIn
            return;
        }

        // Quantize the tensor
        quantizer.Quantize(data, out var quantized, out var scale, out var zeroPoint);

        // Update the tensor data
        tensor.data = quantized;
        var metadata = this.MetadataOf(key);
        metadata[QuantizationMethodKey] = quantizer.GetType().Name;
        metadata[QuantizationScaleKey] = scale.ToString();
        metadata[QuantizationZeroPointKey] = zeroPoint.ToString();
    }

    /// <summary>
    /// Dequantize all compatible tensors in the safetensors set using the given quantizer.
    /// Only tensors of the type TOut will be dequantized back into their original type of TIn and the others will be skipped.
    /// Additionally, the tensor metadata must contain the quantization scale and zero-point or it will be skipped.
    /// </summary>
    /// <typeparam name="TIn">Tensor dequantized type</typeparam>
    /// <typeparam name="TOut">Quantized tensor output type</typeparam>
    /// <param name="quantizer">Quantization method</param>
    public void DequantizeAll<TIn, TOut>(IQuantization<TIn, TOut> quantizer) {
        foreach (var key in this.Keys()) {
            Dequantize(key, quantizer);
        }
    }

    /// <summary>
    /// Dequantize the given tensor in the safetensors set using the given quantizer.
    /// Only tensors of the type TOut will be dequantized back into their original type of TIn. If the tensor is not of type TOut this method does nothing.
    /// Additionally, the tensor metadata must contain the quantization scale and zero-point or this method will do nothing.
    /// </summary>
    /// <typeparam name="TIn">Tensor dequantized type</typeparam>
    /// <typeparam name="TOut">Quantized tensor output type</typeparam>
    /// <param name="quantizer">Quantization method</param>
    public void Dequantize<TIn, TOut>(string key, IQuantization<TIn, TOut> quantizer) {
        var tensor = this.tensors[key];
        if (tensor.data is not TOut[] data) {
            // This quantizer only works for tensors of the type TIn
            return;
        }
        var metadata = this.MetadataOf(key);
        if (!metadata.TryGetValue(QuantizationScaleKey, out string? scaleString) || !metadata.TryGetValue(QuantizationZeroPointKey, out string? zeroPointString)) {
            // This tensor is missing quantization metadata
            return;
        }
        if (!double.TryParse(scaleString, out var scale)) {
            scale = 1.0; // Default scale
        }
        if (!double.TryParse(zeroPointString, out var zeroPoint)) {
            zeroPoint = 0.0; // Default zero-point
        }
        
        // Quantize the tensor
        quantizer.Dequantize(out var dequantized, data, scale, zeroPoint);

        // Update the tensor data
        tensor.data = dequantized;
        metadata.Remove(QuantizationMethodKey);
        metadata.Remove(QuantizationScaleKey);  
        metadata.Remove(QuantizationZeroPointKey);
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
                var shape       = new Shape(tensorInfo.shape ?? []);

                var dimensions  = shape.Rank;
                var entries     = shape.LogicalElementCount();
                var startIndex  = (ulong)tensorInfo.DataStartOffset();
                var endIndex    = (ulong)tensorInfo.DataEndOffset();


                if (endIndex < startIndex) {
                    throw new FormatException($"Tensor {key} has invalid data offsets [{startIndex}, {endIndex}].");
                }

                // Allocate storage
                var data        = make_array(tensorInfo.dtype, entries);
                var type        = data.GetType().GetElementType() ?? typeof(object);
                var matrix_byte_size = (ulong)(Marshal.SizeOf(type) * shape.LogicalElementCount());
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

                        case BFloat16[] bf16_data: bf16_data[i] = new BFloat16(reader.ReadUInt16()); base_stream_pos += sizeof(UInt16); break;

                        default:
                            throw new ArgumentException($"Type {data.GetType()} is not supported");
                    }
                }
                sb.tensors.Add(
                    key, 
                    new UnknownObjectTensor(shape, data)
                );
                if (tensorInfo.__metadata__ is not null && tensorInfo.__metadata__.Count > 0) {
                    sb.tensor_metadata.Add(key, tensorInfo.__metadata__);
                }
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
        } else if (t == "BF16") {
            return new BFloat16[count];
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
        } else if (t == typeof(BFloat16)) {
            return "BF16";
        }

        else {
            throw new ArgumentException($"Type {t} is not supported");
        }
    }

    private static void create_header_for(Dictionary<string, TensorInfo> header, ref ulong buffer_offset, Dictionary<string, UnknownObjectTensor> tensors, Dictionary<string, Dictionary<string, string>> metas) {
        foreach (var matrix in tensors) {
            var shape = matrix.Value.shape;
            var type = matrix.Value.data?.GetType()?.GetElementType() ?? typeof(object);
            var matrix_byte_size = (ulong)(Marshal.SizeOf(type) * shape.LogicalElementCount());
            var metadata = metas.ContainsKey(matrix.Key) ? metas[matrix.Key] : new Dictionary<string, string>();
            header.Add(
                matrix.Key,
                new TensorInfo { 
                    dtype           = type_to_string(type),
                    shape           = shape.AsDimensionSpan().ToArray(),
                    data_offsets    = new ulong[]{ buffer_offset, buffer_offset + matrix_byte_size },
                    __metadata__    = metadata
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
        create_header_for(header, ref buffer_offset, tensors, tensor_metadata);
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

                case BFloat16[] bf16_data: foreach (var v in bf16_data) { writer.Write(v.RawValue); } break;

                default:
                    throw new ArgumentException($"Type {tensor.data.GetType()} is not supported");
            }
        }
    }
}