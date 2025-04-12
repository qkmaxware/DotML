using CommandLine;
using DotML.Network.IO;
using DotML.Network.IO.Netbuild;

namespace DotML.Cli.Commands;

[Verb("build", HelpText = "Compile a netbuild script into a network model")]
public class Build : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "File path to model to build", Required = true)]
    public string? FilePath {get; set;}

    [Option("description", Required = false, HelpText = "Text to use as the model's description")]
    public string? DescriptionText {get; set;}

    [Option("tag", HelpText = "Tag to use to uniquely identify this network once built", Required = false)]
    public IEnumerable<string>? TagsToAdd {get; set;}

    [Option("labels", Required = false, HelpText = "Labels for all output classes", Separator = ' ')]
    public IEnumerable<string>? LabelsForClasses {get; set;}

    public override void Action(AppData appData) {
        FileInfo file;
        if (string.IsNullOrEmpty(FilePath) || !((file = new FileInfo(FilePath)).Exists)) {
            Console.WriteLine($"The file '{FilePath}' doesn't exist.");
            return;
        }

        NetbuildSerializer builder = new NetbuildSerializer();
        Console.Write($"Loading build context {DataSize.FromValue(file.Length)}...");
        var model_scripts = appData.ListModels();
        var scoped_networks = model_scripts
            .SelectMany(model => (model.Tags ?? Enumerable.Empty<string>()).Prepend(model.Guid).Cast<string>().Select(key => new KeyValuePair<string, ModelInfo>(key, model)))
            .GroupBy(x => x.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(
                (model) => model.Key, 
                (model) => (Func<string>)(model.First().Value.GetBuildScript)
            );
        using var reader = new StreamReader(file.OpenRead());
        var text = reader.ReadToEnd();
        Console.WriteLine("done");
        Console.WriteLine();

        Console.Write($"Parsing build commands...");
        var ast = builder.Parse(text);
        Console.WriteLine("done");
        Console.WriteLine();

        Console.WriteLine($"Building network...");
        var network = ast.Make(
            new BuildEnvironment {
                Serializer = builder,
                ScopedNetworks = scoped_networks
            },
            (index, count, statement) => {
                Console.WriteLine($"Step {index + 1}/{count} : {statement}");
            }
        );
        Console.WriteLine();

        Console.Write($"Validating network...");
        bool is_valid = false;
        Exception? error = null;
        try {
            network.ValidateSizes();
            is_valid = true;
        } catch (Exception e) {
            is_valid = false;
            error = e;
        }
        Console.WriteLine("done");
        Console.WriteLine();

        var guid = Guid.NewGuid().ToString();
        var name = TagsToAdd?.FirstOrDefault() ?? network.Name;

        if (is_valid) {
            var network_file = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + ".netbuild"));
            using (var writer = new StreamWriter(network_file.FullName)) {
                builder.Serialize(network, writer);
            }

            var meta_file = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + ".xml"));
            using (var writer = new StreamWriter(meta_file.FullName)) {
                var info = new ModelInfo(meta_file);
                info.Description = DescriptionText;

                if (TagsToAdd is not null) {
                    info.Tags = new List<string>();
                    foreach (var tag in TagsToAdd) {
                        info.Tags.Add(tag);
                    }
                    if (!string.IsNullOrEmpty(network.Name) && !info.Tags.Contains(network.Name)) {
                        info.Tags.Add(network.Name);
                    }
                }
                if (LabelsForClasses is not null) {
                    info.ClassLabels = new List<string>();
                    foreach (var label in LabelsForClasses) {
                        info.ClassLabels.Add(label);
                    }
                }
                writer.Write(info.ToXml());
            }
            Console.WriteLine($"Successfully built model {guid}.");
            if (name is not null)
                Console.WriteLine($"Successfully tagged model {guid} as '{name}'");
        } else {
            Console.WriteLine($"The model is invalid '{error?.Message}'. Please review your build script and try again.");
        }
    }       
}