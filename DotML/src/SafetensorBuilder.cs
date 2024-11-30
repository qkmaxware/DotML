using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace DotML;

/// <summary>
/// Any object that is serializable to safetensor format
/// </summary>
public interface ISafetensorable {
    /// <summary>
    /// Output this network's configuration in the safetensor format
    /// </summary>
    /// <param name="writer">binary writer to write to</param>
    public void ToSafetensor(BinaryWriter writer);

    /// <summary>
    /// Output this network's configuration in the safetensor format
    /// </summary>
    /// <returns>safetensor bytes</returns>
    public byte[] ToSafetensor() {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        ToSafetensor(writer);

        return stream.ToArray();
    }
}

/// <summary>
/// Class to aid in the creation of safetensors files
/// </summary>
public class SafetensorBuilder {
    private HashSet<string> used_names = new HashSet<string>();
    private Dictionary<string, Matrix<Half>> matrices_f16 = new Dictionary<string, Matrix<Half>>();
    private Dictionary<string, Matrix<float>> matrices_f32 = new Dictionary<string, Matrix<float>>();
    private Dictionary<string, Matrix<double>> matrices_f64 = new Dictionary<string, Matrix<double>>();

    /// <summary>
    /// All keys in this safetensors set
    /// </summary>
    /// <returns>enumerable of keys</returns>
    public IEnumerable<string> Keys() { foreach (var key in used_names) { yield return key; }}

    /// <summary>
    /// Check if this safetensors set contains a tensor with the given name
    /// </summary>
    /// <param name="key">key name</param>
    /// <returns>true if key exists</returns>
    public bool ContainsKey(string key) => used_names.Contains(key);

    /// <summary>
    /// Get the tensor associated with the given key
    /// </summary>
    /// <typeparam name="TOut">output type</typeparam>
    /// <param name="key">tensor key</param>
    /// <returns>matrix</returns>
    /// <exception cref="KeyNotFoundException">thrown when the given key doesn't exist in the safetensors set</exception>
    public Matrix<TOut> GetTensor<TOut>(string key) where TOut:INumber<TOut>,IExponentialFunctions<TOut>,IRootFunctions<TOut> {
        if (matrices_f16.TryGetValue(key, out Matrix<Half> m16)) {
            return m16.Transform<TOut>((v) => (TOut)Convert.ChangeType(v, typeof(TOut)));
        }
        if (matrices_f32.TryGetValue(key, out Matrix<float> m32)) {
            return m32.Transform<TOut>((v) => (TOut)Convert.ChangeType(v, typeof(TOut)));
        }
        if (matrices_f64.TryGetValue(key, out Matrix<double> m64)) {
            return m64.Transform<TOut>((v) => (TOut)Convert.ChangeType(v, typeof(TOut)));
        }

        throw new KeyNotFoundException(key);
    }

    /// <summary>
    /// Add a 16bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor value</param>
    public void Add(string name, Matrix<Half> matrix) { used_names.Add(name); matrices_f16.Add(name, matrix); }
    /// <summary>
    /// Add a 16bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor value</param>
    public void Add(string name, Vec<Half> vector) { used_names.Add(name); matrices_f16.Add(name, vector.ToColumnMatrix()); }
    /// <summary>
    /// Add a 32bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor value</param>
    public void Add(string name, Matrix<float> matrix) { used_names.Add(name); matrices_f32.Add(name, matrix); }
    /// <summary>
    /// Add a 32bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor value</param>
    public void Add(string name, Vec<float> vector) { used_names.Add(name); matrices_f32.Add(name, vector.ToColumnMatrix()); }
    /// <summary>
    /// Add a 64bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor value</param>
    public void Add(string name, Matrix<double> matrix) { used_names.Add(name); matrices_f64.Add(name, matrix); }
    /// <summary>
    /// Add a 64bit matrix to the safetensor
    /// </summary>
    /// <param name="name">tensor name</param>
    /// <param name="matrix">tensor value</param>
    public void Add(string name, Vec<double> vector) { used_names.Add(name); matrices_f64.Add(name, vector.ToColumnMatrix()); }

    /// <summary>
    /// Read the tensors from the give file
    /// </summary>
    /// <param name="file">file pointer</param>
    public static SafetensorBuilder ReadFromFile(FileInfo file) {
        using var writer = new BinaryReader(file.OpenRead());
        return ReadFrom(writer);
    }

    /// <summary>
    /// Read the tensors from the give file
    /// </summary>
    /// <param name="path">file path</param>
    public static SafetensorBuilder ReadFromFile(string path) {
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
    public static SafetensorBuilder ReadFrom(BinaryReader reader) {
        var header_size = reader.ReadUInt64();
        var header_json = JsonSerializer.Deserialize<Dictionary<string, TensorInfo>>(System.Text.Encoding.UTF8.GetString(reader.ReadBytes((int)header_size)));
        
        SafetensorBuilder sb = new SafetensorBuilder();
        var buffer_offset = reader.BaseStream.Position;

        if (header_json is not null) {
            foreach (var entry in header_json) {
                var key         = entry.Key;
                var tensorInfo  = entry.Value;

                var dimensions  = tensorInfo.shape?.Length ?? 0;
                var entries     = tensorInfo.shape?.Aggregate(1, (a, b) => a * b) ?? 0;
                if (dimensions != 2) {
                    throw new NotSupportedException("This library only supports loading of 2D vectors.");
                }
                var rows        = tensorInfo.shape?.FirstOrDefault() ?? 0;
                var columns     = tensorInfo.shape?.Skip(1)?.FirstOrDefault() ?? 0;
                if (entries != rows * columns) {
                    throw new FormatException("Mismatched rows and columns for tensor.");
                }

                var entrySize   = tensorInfo.dtype?.ToLower() switch {
                    "f16"       => Marshal.SizeOf(default(Half)),
                    "f32"       => sizeof(float),
                    "f64"       => sizeof(double),
                    null        => throw new NullReferenceException("Tensor datatype missing."),
                    _           => throw new NotSupportedException($"Tensor datatype '{tensorInfo.dtype}' is not supported."),
                };
                var tensorSize  = entries * entrySize;
                
                var start = tensorInfo.data_offsets?.FirstOrDefault() ?? -1;
                var end   = tensorInfo.data_offsets?.Skip(1)?.FirstOrDefault() ?? -1;

                if (start < 0 || end < 0 || end != start + tensorSize) {
                    throw new FormatException("Tensor size doesn't match header data.");
                }

                reader.BaseStream.Position = buffer_offset + start;
                switch (tensorInfo.dtype?.ToLower()) {
                    case "f16":
                        Half[,] m16 = new Half[rows, columns];
                        for (var r = 0; r < rows; r++) {
                            for (var c = 0; c < columns; c++) {
                                m16[r, c] = reader.ReadHalf();
                            }
                        }
                        sb.Add(key, Matrix<Half>.Wrap(m16));
                        break;
                    case "f32":
                        float[,] m32 = new float[rows, columns];
                        for (var r = 0; r < rows; r++) {
                            for (var c = 0; c < columns; c++) {
                                m32[r, c] = reader.ReadSingle();
                            }
                        }
                        sb.Add(key, Matrix<float>.Wrap(m32));
                        break;
                    case "f64":
                        double[,] m64 = new double[rows, columns];
                        for (var r = 0; r < rows; r++) {
                            for (var c = 0; c < columns; c++) {
                                m64[r, c] = reader.ReadDouble();
                            }
                        }
                        sb.Add(key, Matrix<double>.Wrap(m64));
                        break;

                    default:
                        throw new NotSupportedException($"Tensor datatype '{tensorInfo.dtype}' is not supported.");
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
        public int[]? data_offsets {get; set;}
        public Dictionary<string, string>? __metadata__ {get; set;}
    }

    private void create_header_for<T>(Dictionary<string, TensorInfo> header, ref int buffer_offset, string type, Dictionary<string, Matrix<T>> tensors) where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> {
        foreach (var matrix in tensors) {
            var matrix_byte_size = Marshal.SizeOf(default(T)) * matrix.Value.Size;
            header.Add(
                matrix.Key,
                new TensorInfo { 
                    dtype           = type ,
                    shape           = new int[]{ matrix.Value.Rows, matrix.Value.Columns },
                    data_offsets    = new int[]{ buffer_offset, buffer_offset + matrix_byte_size },
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
        var buffer_offset = 0;
        create_header_for(header, ref buffer_offset, "F16", matrices_f16);
        create_header_for(header, ref buffer_offset, "F32", matrices_f32);
        create_header_for(header, ref buffer_offset, "F64", matrices_f64);
        var json = JsonSerializer.Serialize(header);
        var header_bytes = System.Text.Encoding.UTF8.GetBytes(json);
        writer.Write(header_bytes.LongLength);
        writer.Write(header_bytes);

        // Tensors
        foreach (var matrix in matrices_f16) {
            var m = matrix.Value;
            foreach (var value in m) {
                writer.Write(value);
            }
        }
        foreach (var matrix in matrices_f32) {
            var m = matrix.Value;
            foreach (var value in m) {
                writer.Write(value);
            }
        }
        foreach (var matrix in matrices_f64) {
            var m = matrix.Value;
            foreach (var value in m) {
                writer.Write(value);
            }
        }
    }
}