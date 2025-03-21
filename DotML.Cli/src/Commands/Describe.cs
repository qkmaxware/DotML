using System.Security.Cryptography.X509Certificates;
using CommandLine;
using DotML.Network;

namespace DotML.Cli.Commands;

[Verb("describe", HelpText = "Describe the details of an compiled model")]
public class Describe : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    public override void Action(AppData appData) {
        var model = appData.GetModel(ModelName);
        if (model is null) {
            Console.WriteLine($"No model exists with name '{ModelName}'.");
            return;
        }

        Console.WriteLine("IDENTIFIERS");
        Console.WriteLine(" | " + (model.Guid ?? "?"));
        foreach (var tag in model.Tags.Select((t, i) => (i, t))) {
            Console.Write(" | ");
            Console.Write('\''); Console.Write(tag.t); Console.Write('\'');
            Console.WriteLine();
        }
        Console.WriteLine();

        if (model.ClassLabels.Any()) {
        Console.WriteLine("OUTPUT-CLASSES");
        foreach (var label in model.ClassLabels.Select((t, i) => (i, t))) {
            Console.Write(" | ");
            Console.Write('\''); Console.Write(label.t); Console.Write('\'');
            Console.WriteLine();
        }
        Console.WriteLine();
        }

        Console.WriteLine("BUILD-SCRIPT");
        foreach (var line in model.GetBuildScript().Split('\n')) {
            Console.Write(" | ");
            Console.WriteLine(line);
        }
        Console.WriteLine();

        Console.WriteLine("TRAINING");
        string[] train_columns = ["STATUS   ", "ACCURACY ", "PRECISION", "RECALL   ", "LOSS             ", "TRAINING-DURATION "];
        int[] train_len = train_columns.Select(str => str.Length).ToArray();
        Console.Write(" | ");
        for (var col = 0; col < train_columns.Length; col++) {
            var name = train_columns[col];
            var len = train_len[col];
            Console.Write(ColumnValue(name, len));
            Console.Write(' ');
        }
        Console.WriteLine();
        Console.Write(" | ");
        Console.Write(ColumnValue(model.Status(), train_len[0])); Console.Write(' ');
        ModelTrainingInfo? trainingMeta;
        if (model.Status() == ModelTrainingStatus.Trained) {
            if ((trainingMeta = model.TrainingMetadata) is not null) {
                Console.Write(ColumnValue(trainingMeta.Accuracy, train_len[1])); Console.Write(' ');
                Console.Write(ColumnValue(trainingMeta.Precision, train_len[2])); Console.Write(' ');
                Console.Write(ColumnValue(trainingMeta.Recall, train_len[3])); Console.Write(' ');
                Console.Write(ColumnValue($"{trainingMeta.AvgLoss:F3} ± {(trainingMeta.MaxLoss - trainingMeta.MinLoss):F3}", train_len[4])); Console.Write(' ');
                Console.Write(ColumnValue(trainingMeta.TrainingDuration, train_len[5]));
            }
        }
        Console.WriteLine();
        Console.WriteLine();

        Console.WriteLine("ARCHITECTURE");
        var network = model.Load();
        string[] arch_columns = ["LAYER-TYPE       ", "INPUT-SHAPE", "OUTPUT-SHAPE", "TRAINABLE-PARAMS", "UNTRAINABLE-PARAMS", "DESCRIPTION"];
        string[] summary_columns = ["LAYERS", "STORAGE-SIZE", "TRAINABLE-PARAMS", "UNTRAINABLE-PARAMS"];
        int[] sum_length = [arch_columns[0].Length, 1 + arch_columns[1].Length + arch_columns[2].Length, arch_columns[3].Length, arch_columns[4].Length];
        Console.Write(" | ");
        for (var col = 0; col < summary_columns.Length; col++) {
            var name = summary_columns[col];
            var len = sum_length[col];
            Console.Write(ColumnValue(name, len));
            Console.Write(' ');
        }
        Console.WriteLine();
        Console.Write(" | ");
        Console.Write(ColumnValue(network.LayerCount, sum_length[0])); Console.Write(' ');
        Console.Write(ColumnValue(network.StorageSize(), sum_length[1])); Console.Write(' ');
        Console.Write(ColumnValue(network.TrainableParameterCount(), sum_length[2])); Console.Write(' ');
        Console.Write(ColumnValue(network.UnTrainableParameterCount(), sum_length[3])); Console.Write(' ');
        Console.WriteLine();

        Console.WriteLine(" | ");
        int[] arch_len = arch_columns.Select(str => str.Length).ToArray();
        Console.Write(" | ");
        for (var col = 0; col < arch_columns.Length; col++) {
            var name = arch_columns[col];
            var len = arch_len[col];
            Console.Write(ColumnValue(name, len));
            Console.Write(' ');
        }
        Console.WriteLine();

        var describer = new LayerDescriber();
        for (var layerIndex = 0; layerIndex < network.LayerCount; layerIndex++) {
            var layer = network.GetLayer(layerIndex);
            Console.Write(" | ");
            Console.Write(ColumnValue(TrimEnd(layer.GetType().Name, "Layer"), arch_len[0]));
            Console.Write(' ');
            Console.Write(ColumnValue(layer.InputShape, arch_len[1]));
            Console.Write(' ');
            Console.Write(ColumnValue(layer.OutputShape, arch_len[2]));
            Console.Write(' ');
            Console.Write(ColumnValue(layer.TrainableParameterCount(), arch_len[3]));
            Console.Write(' ');
            Console.Write(ColumnValue(layer.UnTrainableParameterCount(), arch_len[4]));
            Console.Write(' ');
            Console.Write(layer.Visit(describer), arch_len[5]);
            Console.WriteLine();
        }
        Console.WriteLine();
    }

    private static string TrimEnd(string src, string postfix) {
        if (!src.EndsWith(postfix))
            return src;

        return src.Remove(src.LastIndexOf(postfix));
    }
}