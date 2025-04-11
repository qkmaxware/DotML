using System.Text;
using CommandLine;
using DotML.Cli.Logging;

namespace DotML.Cli.Commands;

// Example usage
/*
dotml run net2 --input img.png --embedding img --decoder probability --output distribution.txt
*/

// Example output
/*
vectorizing "img.png"...done
processing...done
decoding...done created "distribution.txt"

class 0: |-------------| 100%
class 1: |             | 0%


*/

[Verb("run", HelpText = "Run a compiled model against a given input")]
public class Run : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    [Option('f', "input", HelpText = "Path to input files, if no file present stdin is used instead")]
    public string? InputFile {get; set;}

    [Option("embedding", HelpText = "Embedding type", Required = true)]
    public string? EmbeddingType {get; set;}

    [Option("decoder", HelpText = "Decoder type", Required = false, Default = "Vector")]
    public string? DecoderType {get; set;}

    [Option('o', "output", HelpText = "Save model output to a file at the given path", Required = false)]
    public string? OutputFile {get; set;}

    [Option("labels", HelpText = "Labels for classifications/categories", Required = false)]
    public IEnumerable<string>? Labels {get; set;}

    [Option("log", HelpText = "List of items to save to the logs as the network is run (tensors, images, etc.)", Required = false)]
    public IEnumerable<string>? OutputLoggers {get; set;}

    public override void Action(AppData appData) {
        // Verify options
        var assembly = typeof(Run).Assembly;
        var available_embedders = assembly.GetExportedTypes().Where(type => !type.IsAbstract && type.IsAssignableTo(typeof(IEmbedder)));
        var available_decoders  = assembly.GetExportedTypes().Where(type => !type.IsAbstract && type.IsAssignableTo(typeof(IDecoder)));
        IEmbedder? embedder     = available_embedders.Where(type => type.Name.Equals(EmbeddingType, StringComparison.CurrentCultureIgnoreCase)).Select(type => (IEmbedder?)Activator.CreateInstance(type)).FirstOrDefault();
        IDecoder? decoder       = available_decoders.Where(type => !type.IsAbstract && type.IsAssignableTo(typeof(IDecoder))).Where(type => type.Name.Equals(DecoderType, StringComparison.CurrentCultureIgnoreCase)).Select(type => (IDecoder?)Activator.CreateInstance(type)).FirstOrDefault();
        if (embedder is null) {
            Console.Write($"No embedding type with the name '{EmbeddingType}'. Available embeddings include: ");
            Console.Write(string.Join(", ", available_embedders.Select(x => x.Name)));
            Console.WriteLine(".");
            return;
        }
        if (decoder is null) {
            Console.Write($"No decoder type with the name '{DecoderType}'. Available decoders include: ");
            Console.Write(string.Join(", ", available_decoders.Select(x => x.Name)));
            Console.WriteLine(".");
            return;
        }
        var user_selected_output_loggers = OutputLoggers?.ToArray() ?? Array.Empty<string>();
        DirectoryInfo[] log_dir = [new DirectoryInfo(appData.GenerateReportPath("Run"))];
        var output_loggers 
            = assembly.GetExportedTypes().Where(type => !type.IsAbstract && type.IsAssignableTo(typeof(IOutputLogger)))
            .Where(type => user_selected_output_loggers.Contains(type.Name.Replace("OutputLogger", string.Empty), StringComparer.OrdinalIgnoreCase))
            .Select(type => (IOutputLogger?)Activator.CreateInstance(type, log_dir)).ToArray();
        var is_logging = output_loggers.Length > 0;
        
        var model = appData.GetModel(ModelName);
        if (model is null) {
            Console.WriteLine($"No model exists with name '{ModelName}'.");
            return;
        }

        if (decoder is IWithLabels labeled) {
            string[] labels;
            if (this.Labels is not null && this.Labels.Any()) {
                labels = this.Labels.ToArray();
            } else if (model.ClassLabels is not null && model.ClassLabels.Any()) {
                labels = model.ClassLabels.ToArray();
            } else {
                labels = Array.Empty<string>();
            }
            labeled.Labels = labels;
        }

        // Do work
        Console.Write("Loading model...");
        var network = model.Load();
        Console.WriteLine("done");

        Console.Write($"Vectorizing '{(string.IsNullOrEmpty(InputFile) ? "stdin" : InputFile)}'...");
        BatchedFeatureSet<double> input_vector;
        if (string.IsNullOrEmpty(InputFile)) {
            using var reader = new StreamReader(Console.OpenStandardInput(), Console.InputEncoding);
            var input = reader.ReadToEnd();
            input_vector = embedder.CreateEmbedding(network, input);
        } else {
            FileInfo input = new FileInfo(InputFile);
            if (!input.Exists) {
                Console.WriteLine($"No file exists with name '{InputFile}'.");
                return;
            }
            input_vector = embedder.CreateEmbedding(network, new FileInfo[]{ input });
        }
        Console.WriteLine("done");

        Console.Write("Predicting output...");
        var layer_index = 0;
        var output_vector = network.PredictSync(
            values: input_vector,
            before_layer: (layer, input) => {},
            after_layer: (layer, output) => {
                foreach (var output_logger in output_loggers) {
                    if (output_logger is not null)
                        layer.Visit(output_logger, (layer_index, output));
                }
                layer_index++;
            }
        );
        Console.WriteLine("done");

        Console.Write("Decoding...");
        using (var result = decoder.Decode(output_vector)) {
            if (!string.IsNullOrEmpty(OutputFile)) {
                result.FileOutput(new FileInfo(OutputFile));
                Console.WriteLine($"done created '{OutputFile}'");
            } else {
                Console.WriteLine("done");
            }

            if (is_logging) {
                Console.WriteLine();
                Console.WriteLine($"Reports saved to '{log_dir[0].Name}'.");
                Console.WriteLine($"Use \"{typeof(Run).Assembly.GetName().Name} reports open '{log_dir[0].Name}'\" to review runtime logs.");
            }

            DrawDivider();
            result.ConsoleOutput();
        }
    }
}