using DotML.Network.Training;

namespace DotML.Cli.TrainingData;

/// <summary>
/// Treat the training data as a binary classified vector. THIS NEEDS CHANGES EACH TIME TO WORK PROPERLY. TODO CUSTOMIZE FROM CLI ARGS
/// </summary>
public class BinaryClassifiedVectors : ITrainingDataFormat {

    public int OutputClasses = 10;
    public double ZeroValue = 0.0;
    public double OneValue = 1.0;

    public bool IsInFormat(FileInfo file) {
        return file.Extension == ".bin" && !TrainingSet.IsBinaryTrainingSet(file);
    }

    public TrainingSet Read(FileInfo file) {
        return read_classified_binary_vectors(file, OutputClasses, ZeroValue, OneValue, x => x.ReadByte());
    }

    private static Vec<double> vector_from_label_index(int index, int classes, double off = -1, double on = 1) {
        double[] values = new double[classes];
        Array.Fill(values, off);
        if (index >= 0 && index < classes)
            values[index] = on;
        return Vec<double>.Wrap(values);
    }

    private static TrainingSet read_classified_binary_vectors(FileInfo file, int category_count, double category_off, double category_on, Func<BinaryReader, double> element_parser, int? fixed_vector_size = null) {
        using var stream = file.OpenRead();
        using var reader = new BinaryReader(stream);
                            
        List<TrainingPair> pairs = new List<TrainingPair>();
        while (stream.Position < stream.Length) {
            var category_index  = reader.ReadByte();
            var vector_size     = fixed_vector_size.HasValue ? fixed_vector_size.Value : reader.ReadInt32();
            double[] input_vec  = new double[vector_size];

            for (var i = 0; i < vector_size; i++) {
                try {
                    input_vec[i] = element_parser(reader);
                } catch {
                    input_vec[i] = default(double);
                }
            } 
            pairs.Add(new TrainingPair { Input = Vec<double>.Wrap(input_vec), Output = vector_from_label_index(category_index, category_count, category_off, category_on) });
        }

        return new TrainingSet(pairs);
    }
}