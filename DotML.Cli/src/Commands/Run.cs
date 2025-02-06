using System.Text;
using CommandLine;

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

[Verb("run", HelpText = "Run a compiled network against a given input")]
public class Run : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    [Option('f', "input", HelpText = "Path to input file, if no file present stdin is used instead")]
    public string? InputFile {get; set;}

    [Option("embedding", HelpText = "Embedding type", Required = true)]
    public string? EmbeddingType {get; set;}

    [Option("decoder", HelpText = "Decoder type", Required = false, Default = "Vector")]
    public string? DecoderType {get; set;}

    [Option('o', "output", HelpText = "Save model output to a file at the given path", Required = false)]
    public string? OutputFile {get; set;}

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

        var model = appData.GetModel(ModelName);
        if (model is null) {
            Console.WriteLine($"No model exists with name '{ModelName}'.");
            return;
        }

        // Do work
        Console.Write("Loading model...");
        var network = model.Load();
        Console.WriteLine("done");

        Console.Write($"Vectorizing '{(string.IsNullOrEmpty(InputFile) ? "stdin" : InputFile)}'...");
        Vec<double> input_vector;
        if (string.IsNullOrEmpty(InputFile)) {
            StringBuilder sb = new StringBuilder();
            while (Console.In.Peek() != -1) {
                sb.Append(Console.In.ReadLine());
            }
            input_vector = embedder.CreateEmbedding(sb.ToString());
        } else {
            FileInfo input = new FileInfo(InputFile);
            if (!input.Exists) {
                Console.WriteLine($"No file exists with name '{InputFile}'.");
                return;
            }
            input_vector = embedder.CreateEmbedding(input);
        }
        Console.WriteLine("done");

        Console.Write("Predicting output...");
        var output_vector = network.PredictSync(input_vector);
        Console.WriteLine("done");

        Console.Write("Decoding...");
        using (var result = decoder.Decode(network.OutputShape, output_vector)) {
            if (!string.IsNullOrEmpty(OutputFile)) {
                result.FileOutput(new FileInfo(OutputFile));
                Console.WriteLine($"done created '{OutputFile}'");
            } else {
                Console.WriteLine("done");
            }
            DrawDivider();
            result.ConsoleOutput();
        }
    }
}