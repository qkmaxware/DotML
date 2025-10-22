using System.Text;
using CommandLine;
using DotML.Cli.Logging;
using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

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

    [Option("log", HelpText = "List of items to save to the logs as the network is run (tensors, images, stats, etc.)", Required = false)]
    public IEnumerable<string>? OutputLoggers {get; set;}

    private enum TaskState
    {
        Waiting, Running, Done
    }

    private IElement MakeTask(Func<TaskState> condition, string name)
    {
        return new VBox(
            new Conditional(
                () => condition() == TaskState.Waiting,
                new Label(name)
            ),
            new Conditional(
                () => condition() == TaskState.Running,
                new Spinner(CharacterAnimation.TravellingDots, name  + "...")
            ),
            new Conditional(
                () => condition() == TaskState.Done,
                new Label("✓ " + name)
            )
        );
    }

    public override void Action(AppData appData) {
        var rendering = true;
        Task? renderTask = null;
        try
        {
            // Verify options
            var assembly = typeof(Run).Assembly;
            var available_embedders = assembly.GetExportedTypes().Where(type => !type.IsAbstract && type.IsAssignableTo(typeof(IEmbedder)));
            var available_decoders = assembly.GetExportedTypes().Where(type => !type.IsAbstract && type.IsAssignableTo(typeof(IDecoder)));
            IEmbedder? embedder = available_embedders.Where(type => type.Name.Equals(EmbeddingType, StringComparison.CurrentCultureIgnoreCase)).Select(type => (IEmbedder?)Activator.CreateInstance(type)).FirstOrDefault();
            IDecoder? decoder = available_decoders.Where(type => !type.IsAbstract && type.IsAssignableTo(typeof(IDecoder))).Where(type => type.Name.Equals(DecoderType, StringComparison.CurrentCultureIgnoreCase)).Select(type => (IDecoder?)Activator.CreateInstance(type)).FirstOrDefault();
            if (embedder is null)
            {
                WriteError(
                    $"No embedding type with the name '{EmbeddingType}'.",
                    available_embedders.Select(x => x.Name)
                );
                return;
            }
            if (decoder is null)
            {
                WriteError(
                    $"No decoder type with the name '{DecoderType}'.",
                    available_decoders.Select(x => x.Name)
                );
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
            if (model is null)
            {
                WriteError($"No model exists with name '{ModelName}'.");
                return;
            }

            if (decoder is IWithLabels labeled)
            {
                string[] labels;
                if (this.Labels is not null && this.Labels.Any())
                {
                    labels = this.Labels.ToArray();
                }
                else if (model.ProblemDescription?.Classification?.ClassLabels is not null)
                {
                    labels = model.ProblemDescription.Classification.ClassLabels.ToArray();
                }
                else
                {
                    labels = Array.Empty<string>();
                }
                labeled.Labels = labels;
            }

            var process_list = new UnorderedList();
            var process_panel = new Panel("Process", process_list);
            var view_stack = new VBox(process_panel);
            var view = new RenderView(view_stack);

            var loadModelTaskState = TaskState.Waiting;
            var loadModelTaskUi = MakeTask(() => loadModelTaskState, "Load Model");
            process_list.Add(loadModelTaskUi);

            var vectorizingTaskState = TaskState.Waiting;
            var vectorizingTaskUi = MakeTask(() => vectorizingTaskState, $"Embedding '{(string.IsNullOrEmpty(InputFile) ? "stdin" : InputFile)}'");
            process_list.Add(vectorizingTaskUi);

            var predictingTaskState = TaskState.Waiting;
            var predictingTaskUi = MakeTask(() => predictingTaskState, "Predicting");
            process_list.Add(predictingTaskUi);

            var decodingTaskState = TaskState.Waiting;
            var decodingTaskUi = MakeTask(() => decodingTaskState, "Decoding Output");
            process_list.Add(decodingTaskUi);


            renderTask = Task.Run(() => view.RenderWhile((self) => rendering)); // Render loop running async

            // Do work
            loadModelTaskState = TaskState.Running;
            var network = model.Load();
            loadModelTaskState = TaskState.Done;

            vectorizingTaskState = TaskState.Running;
            BatchedFeatureSet<float> input_vector;
            if (string.IsNullOrEmpty(InputFile))
            {
                using var reader = new StreamReader(Console.OpenStandardInput(), Console.InputEncoding);
                var input = reader.ReadToEnd();
                input_vector = embedder.CreateEmbedding(network, input);
            }
            else
            {
                FileInfo input = new FileInfo(InputFile);
                if (!input.Exists)
                {
                    rendering = false;
                    renderTask.Wait();
                    WriteError($"No file exists with name '{InputFile}'.");
                    return;
                }
                input_vector = embedder.CreateEmbedding(network, new FileInfo[] { input });
            }
            vectorizingTaskState = TaskState.Done;

            predictingTaskState = TaskState.Running;
            var layer_index = 0;
            var output_vector = network.PredictSync(
                values: input_vector,
                before_layer: (layer, input) => { },
                after_layer: (layer, output) =>
                {
                    foreach (var output_logger in output_loggers)
                    {
                        if (output_logger is not null)
                            layer.Visit(output_logger, (layer_index, output));
                    }
                    layer_index++;
                }
            );
            predictingTaskState = TaskState.Done;


            decodingTaskState = TaskState.Running;
            using (var result = decoder.Decode(output_vector))
            {
                decodingTaskState = TaskState.Done;
                rendering = false;
                renderTask.Wait();

                var output_stack = new VBox();
                var output_panel = new Panel("Output", output_stack);
                if (string.IsNullOrEmpty(OutputFile) && decoder is IFileOnlyDecoder fonly)
                {
                    // No file is provided, but the decoder is a file only decoder
                    if (fonly.FileRequired())
                    {
                        WriteError($"Decoder '{fonly.GetType().Name}' REQUIRES a file to be specified for it's output. Please provide an output file path using the -o or --output options and try again.");
                        return; // Don't even do console output and just hard quit right now
                    }
                    else
                    {
                        output_stack.Add(new Paragraph($"Decoder '{fonly.GetType().Name}' works best when a file is specified for it's output. Results may be displayed on the console but may be insufficient. You may run the network again using the -o or --output options to obtain output files."));
                    }
                }

                output_stack.Add(result.ConsoleOutput());
                view_stack.Add(output_panel);

                if (!string.IsNullOrEmpty(OutputFile))
                {
                    foreach (var file in result.FileOutput(new FileInfo(OutputFile)))
                    {
                        output_stack.Add(new Label($"Created file '{file.Name}'"));
                    }
                }

                if (is_logging)
                {
                    var logging_stack = new VBox();
                    var logging_panel = new Panel("Logs", logging_stack);
                    view_stack.Add(logging_panel);

                    logging_stack.Add(new Label($"Reports saved to '{log_dir[0].Name}'."));
                    logging_stack.Add(new Paragraph($"Use \"{typeof(Run).Assembly.GetName().Name} reports open '{log_dir[0].Name}'\" to review runtime logs."));
                }

                view.RenderOnce();
            }
        } catch (Exception)
        {
            // Make sure all rendering is done.
            rendering = false;
            renderTask?.Wait();
            throw;
        }
    }
}