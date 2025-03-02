using System.Text.Json;
using DotML.Network.Training;

namespace DotML.Cli.TrainingData;

/// <summary>
/// Treat the training data as a classified CSV in the format of [class_number, vector0, ..., vectorN]
/// </summary>
public class ClassifiedCsv : ITrainingDataFormat {

    public bool IsInFormat(FileInfo file) {
        return file.Extension == ".csv";
    }

    private static bool isHeader(string line) {
        return line.Contains("class", StringComparison.CurrentCultureIgnoreCase) || line.Contains("label", StringComparison.CurrentCultureIgnoreCase);
    }

    public TrainingSet Read(FileInfo file) {
        List<(Vec<double>, int)> items = new List<(Vec<double>, int)>();
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
                double.TryParse(x, out double res);
                return res;
            }).ToArray();

            var class_label = (int)data[0];
            var vector_data = data[1..].ToArray();
            max_count = Math.Max(max_count, class_label + 1);
            items.Add((vector_data, class_label));
            first_row = false;
        }

        return new TrainingSet(items.Select(item=> new TrainingPair { Input=item.Item1, Output=vector_from_label_index(item.Item2, max_count, 0, 1) }));
    }

    private static Vec<double> vector_from_label_index(int index, int classes, double off = -1, double on = 1) {
        double[] values = new double[classes];
        Array.Fill(values, off);
        if (index >= 0 && index < classes)
            values[index] = on;
        return Vec<double>.Wrap(values);
    }
}