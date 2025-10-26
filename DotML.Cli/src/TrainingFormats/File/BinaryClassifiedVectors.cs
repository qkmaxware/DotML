using DotML.Network.Training;

namespace DotML.Cli.TrainingData;

/// <summary>
/// Treat the training data as a binary classified vector. THIS NEEDS CHANGES EACH TIME TO WORK PROPERLY. TODO CUSTOMIZE FROM CLI ARGS
/// </summary>
public class BinaryClassifiedVectors : FileTrainingDataFormat {

    public int OutputClasses = 10;
    public float ZeroValue = 0.0f;
    public float OneValue = 1.0f;

    public override bool IsInFormat(FileInfo file) {
        return file.Extension == ".bin" && !TrainingSet<float>.IsBinaryTrainingSet(file);
    }

    private static Vec<float> vector_from_label_index(int index, int classes, float off = -1, float on = 1) {
        float[] values = new float[classes];
        Array.Fill(values, off);
        if (index >= 0 && index < classes)
            values[index] = on;
        return Vec<float>.Wrap(values);
    }

    private static TrainingSet<float> read_classified_binary_vectors(FileInfo file, float category_off, float category_on, Func<BinaryReader, double> element_parser, int? fixed_vector_size = null) {
        using var stream = file.OpenRead();
        using var reader = new BinaryReader(stream);
        
        List<(Vec<float>, int)> items = new List<(Vec<float>, int)>();
        int category_count = 1;
        while (stream.Position < stream.Length) {
            var category_index  = reader.ReadByte();
            category_count = Math.Max(category_count, category_index + 1);
            var vector_size     = fixed_vector_size.HasValue ? fixed_vector_size.Value : reader.ReadInt32();
            float[] input_vec  = new float[vector_size];

            for (var i = 0; i < vector_size; i++) {
                try {
                    input_vec[i] = (float)element_parser(reader);
                } catch {
                    input_vec[i] = default(float);
                }
            } 
            items.Add((Vec<float>.Wrap(input_vec), category_index));
        }
        
        return new TrainingSet<float>(items.Select(item => new TrainingPair<float> { Input=item.Item1, Output=vector_from_label_index(item.Item2, category_count, category_off, category_on) }));
    }

    public override ITrainingDataSource<float> Read(FileInfo file)
    {
        var set = read_classified_binary_vectors(file, ZeroValue, OneValue, x => x.ReadByte());
        
        var src = new ListTrainingDataSource<float>(new TensorShape(set.First().Input.Dimensionality), new TensorShape(set.First().Output.Dimensionality));
        src.AddRange(
            set.Select(pair => (Tensor<float>.Vec(pair.Input.AsArray()), Tensor<float>.Vec(pair.Output.AsArray())))
        );
        return src;
    }
}