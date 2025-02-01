using CommandLine;

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

        Console.WriteLine("GUID");
        Console.WriteLine(" | " + (model.Guid ?? "?"));
        Console.WriteLine();

        Console.WriteLine("TAGS");
        Console.Write(" | ");
        foreach (var tag in model.Tags.Select((t, i) => (i, t))) {
            if (tag.i != 0)
                Console.Write(", ");
            Console.Write('\''); Console.Write(tag.t); Console.Write('\'');
        }
        Console.WriteLine();
        Console.WriteLine();

        Console.WriteLine("BUILD-SCRIPT");
        foreach (var line in model.GetBuildScript().Split('\n')) {
            Console.Write(" | ");
            Console.WriteLine(line);
        }
        Console.WriteLine();

        Console.WriteLine("TRAINING");
        Console.Write(" | ");
        Console.WriteLine(model.Status());
        Console.WriteLine();

        Console.WriteLine("ARCHITECTURE");
        string[] columns = ["INPUT-SHAPE", "OUTPUT-SHAPE", "TRAINABLE-PARAMS", "UNTRAINABLE_PARAMS", "DESCRIPTION"];
        int[] lengths = columns.Select(str => str.Length).ToArray();
        lengths[^1] = 80;
        Console.Write(" | ");
        for (var col = 0; col < columns.Length; col++) {
            var name = columns[col];
            var len = lengths[col];
            Console.Write(ColumnValue(name, len));
            Console.Write(' ');
        }
        Console.WriteLine();

        var network = model.Load();
        for (var layerIndex = 0; layerIndex < network.LayerCount; layerIndex++) {
            var layer = network.GetLayer(layerIndex);
            Console.Write(" | ");
            Console.Write(ColumnValue(layer.InputShape, lengths[0]));
            Console.Write(' ');
            Console.Write(ColumnValue(layer.OutputShape, lengths[1]));
            Console.Write(' ');
            Console.Write(ColumnValue(layer.TrainableParameterCount(), lengths[2]));
            Console.Write(' ');
            Console.Write(ColumnValue(layer.UnTrainableParameterCount(), lengths[3]));
            Console.Write(' ');
            Console.Write(ColumnValue("", lengths[4])); // TODO description like from the markdown exporter
            Console.WriteLine();
        }
        Console.WriteLine();
    }
}