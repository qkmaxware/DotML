using System.Text.Json;
using DotML.Network.Training;

namespace DotML.Cli.TrainingData;

/// <summary>
/// Treat the training data as a classified CSV in the format of [class_number, vector0, ..., vectorN]
/// </summary>
public class ClassifiedCsv : FileTrainingDataFormat {

    public override bool IsInFormat(FileInfo file) {
        return file.Extension == ".csv";
    }

    private static bool isHeader(string line) {
        return line.Contains("class", StringComparison.CurrentCultureIgnoreCase) || line.Contains("label", StringComparison.CurrentCultureIgnoreCase);
    }

    public override ITrainingDataSource<float> Read(FileInfo file) {
        List<(Vec<float>, int)> items = new List<(Vec<float>, int)>();
        var max_count = 1;
        using var reader = new StreamReader(file.OpenRead());
        bool first_row = true;
        string? line;
        while ((line = reader.ReadLine()) is not null) {
            if (first_row && isHeader(line)) {
                // Skip the header
                first_row = false;
            }

            var data = line.Split(',').Select(x => {
                float.TryParse(x, out float res);
                return res;
            }).ToArray();

            var class_label = (int)data[0];
            var vector_data = data[1..].ToArray();
            max_count = Math.Max(max_count, class_label + 1);
            items.Add((vector_data, class_label));
            first_row = false;
        }

        return new ListTrainingDataSource<float>(
            new Shape(),
            new Shape(max_count),
            items.Select(item =>
                (
                    Tensor<float>.Vec(item.Item1.AsArray()),
                    Tensor<float>.Vec(vector_from_label_index(item.Item2, max_count, 0, 1).AsArray())
                )
            )
        );
    }

    private static Vec<float> vector_from_label_index(int index, int classes, float off = -1, float on = 1) {
        float[] values = new float[classes];
        Array.Fill(values, off);
        if (index >= 0 && index < classes)
            values[index] = on;
        return Vec<float>.Wrap(values);
    }
}