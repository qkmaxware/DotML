using System.Collections.ObjectModel;
using System.Numerics;

namespace DotML.Network.Training.Formats;

/// <summary>
/// <para>
/// Binary encoded tensors. Similar to <see cref="BinaryVectorLoader"/> but support for arbitrary shaped tensors.
/// </para>
/// <para>
/// Spec:
/// <code>
/// file := header, {output: 0..outputs}, {input: 0..inputs}
/// header := 't', 'e', 'n', dtype[u8], scaling[f64], outputs[i32], shape, inputs[i32], shape;
/// output := {element[dtype]: 0..size};
/// input := outputPtr[i32], {element[dtype]: 0..size};
/// shape := rank[i32], {length[int]: 0..rank};
/// </code>
/// <para>
/// </summary>
/// <typeparam name="T"></typeparam>
public class BinaryTensorLoader<T> : ITrainingDataLoader<T>
where T : INumber<T>
{
    private static char[] magic = ['t', 'e', 'n'];
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
        int output_rank = reader.ReadInt32();
        var output_lengths = new int[output_rank];
        for (var i = 0; i < output_rank; i++)
            output_lengths[i] = reader.ReadInt32();
        var output_shape = new Shape(output_lengths);

        var input_count = reader.ReadInt32();
        int input_rank = reader.ReadInt32();
        var input_lengths = new int[input_rank];
        for (var i = 0; i < output_rank; i++)
            input_lengths[i] = reader.ReadInt32();
        var input_shape = new Shape(input_lengths);

        ListTrainingDataSource<T> src = new ListTrainingDataSource<T>(input_shape, output_shape);

        // Outputs
        var outputs = new List<Tensor<T>>(output_count);
        for (var i = 0; i < output_count; i++)
        {
            var vec_size = output_shape.LogicalElementCount();
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
            outputs.Add(Tensor<T>.FromFlattenedArray(output_shape, data));
        }

        for (var i = 0; i < input_count; i++)
        {
            var output_index = reader.ReadInt32();
            var vec_size = input_shape.LogicalElementCount();
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
            var input = Tensor<T>.FromFlattenedArray(input_shape, data);
            var output = outputs[output_index];
            src.Add((input, output));
        }

        return src;
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
        writer.Write(src.OutputShape.Rank);         // Set output shape
        for (var i = 0; i < src.OutputShape.Rank; i++)
            writer.Write(src.OutputShape.Length(i));
        writer.Write(src.Count);                    // Set input count
        writer.Write(src.InputShape.Rank);          // Set input shape
        for (var i = 0; i < src.InputShape.Rank; i++)
            writer.Write(src.InputShape.Length(i));

        // Outputs
        foreach (var output in outputs)
        {
            foreach (var element in output.AsSpan())
            {
                writer.Write((double)Convert.ChangeType(element, typeof(double)));
            }
        }

        // Inputs
        foreach (var pair in src) {
            var output_index = Array.IndexOf(outputs, pair.Output);
            writer.Write(output_index);
            foreach (var element in pair.Input.AsSpan()) {
                writer.Write((double)Convert.ChangeType(element, typeof(double)));
            }
        }
    }
}