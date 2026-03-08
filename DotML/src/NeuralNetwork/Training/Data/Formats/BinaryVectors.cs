using System.Collections.ObjectModel;
using System.Numerics;

namespace DotML.Network.Training.Formats;

/// <summary>
/// An object that can load training data from a given source
/// </summary>
/// <typeparam name="T">data format</typeparam>
public interface ITrainingDataLoader<T>
where T : INumber<T>
{
    public ITrainingDataSource<T> Load(string path);
    public void SaveFile(string path, ITrainingDataSource<T> src);
}

/// <summary>
/// <para>
/// Binary encoded vectors
/// </para>
/// <para>
/// Spec:
/// <code>
/// file := header, {output: 0..outputs}, {input: 0..inputs}
/// header := 'v', 'e', 'c', dtype[u8], scaling[f64], outputs[i32], inputs[i32];
/// output := size[i32], {element[dtype]: 0..size};
/// input := outputPtr[i32], size[i32], {element[dtype]: 0..size};
/// </code>
/// <para>
/// </summary>
/// <typeparam name="T"></typeparam>
public class BinaryVectorLoader<T>: ITrainingDataLoader<T>
where T:INumber<T>
{
    private static char[] magic = ['v', 'e', 'c'];
    internal static ReadOnlyCollection<char> BinaryTrainingSetMagicNumber => Array.AsReadOnly(magic);

    public ITrainingDataSource<T> Load(string path)
    {
        using var reader = new BinaryReader(File.OpenRead(path));

        foreach (var magic in BinaryTrainingSetMagicNumber)
        {
            if (reader.ReadByte() != magic)
                throw new FormatException("Stream is not formatted as a binary training set");
        }

        var type = (TrainingVectorStorageType)(reader.ReadByte());
        var scaling = reader.ReadDouble();
        var output_count = reader.ReadInt32();
        var input_count = reader.ReadInt32();

        ListTrainingDataSource<T>? src = null;

        // Outputs
        var outputs = new List<Tensor<T>>(output_count);
        for (var i = 0; i < output_count; i++)
        {
            var vec_size = reader.ReadInt32();
            var data = new T[vec_size];
            for (var j = 0; j < vec_size; j++)
            {
                var raw = type switch
                {
                    TrainingVectorStorageType.U8 => (double)reader.ReadByte(),
                    TrainingVectorStorageType.U16 => (double)reader.ReadUInt16(),
                    TrainingVectorStorageType.U32 => (double)reader.ReadUInt32(),
                    TrainingVectorStorageType.U64 => (double)reader.ReadUInt64(),

                    TrainingVectorStorageType.I8 => (double)reader.ReadSByte(),
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
            outputs.Add(Tensor<T>.Vec(data));
        }

        for (var i = 0; i < input_count; i++)
        {
            var output_index = reader.ReadInt32();
            var vec_size = reader.ReadInt32();
            var data = new T[vec_size];
            for (var j = 0; j < vec_size; j++)
            {
                var raw = type switch
                {
                    TrainingVectorStorageType.U8 => (double)reader.ReadByte(),
                    TrainingVectorStorageType.U16 => (double)reader.ReadUInt16(),
                    TrainingVectorStorageType.U32 => (double)reader.ReadUInt32(),
                    TrainingVectorStorageType.U64 => (double)reader.ReadUInt64(),

                    TrainingVectorStorageType.I8 => (double)reader.ReadSByte(),
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
            var input = Tensor<T>.Vec(data);
            var output = outputs[output_index];
            if (src is null)
                src = new ListTrainingDataSource<T>(new Shape(input.ElementCount), new Shape(output.ElementCount));
            src.Add((input, output));
        }

        return src ?? new ListTrainingDataSource<T>(new Shape(), new Shape());
    }

    public void SaveFile(string path, ITrainingDataSource<T> src) {
        using var writer = new BinaryWriter(File.Open(path, FileMode.Create));

        // Write magic number
        foreach (var magic in BinaryTrainingSetMagicNumber) {
            writer.Write((byte)magic);
        }

        // Compute number of unique outputs
        var outputs = src.Select(pair => pair.Output).Distinct().ToArray();
        // Compute number of unique inputs
        // Compute vector "scaling" factor
        const double scaling = 1.0; // Assume that scaling was already applied

        // DATA_TYPE SCALING OUT_CLASSES, INPUT_CLASSES
        writer.Write((byte)TrainingVectorStorageType.F64);  // Always write F64
        writer.Write(scaling);                      // Set scaling factor
        writer.Write(outputs.Length);               // Set output count
        writer.Write(src.Count);                    // Set input count

        // Outputs
        foreach (var output in outputs) {
            writer.Write(output.ElementCount);
            foreach (var element in output.AsSpan()) {
                writer.Write((double)Convert.ChangeType(element, typeof(double)));
            }
        }

        // Inputs
        foreach (var pair in src) {
            var output_index = Array.IndexOf(outputs, pair.Output);
            writer.Write(output_index);
            writer.Write(pair.Input.ElementCount);
            foreach (var element in pair.Input.AsSpan()) {
                writer.Write((double)Convert.ChangeType(element, typeof(double)));
            }
        }
    }
}