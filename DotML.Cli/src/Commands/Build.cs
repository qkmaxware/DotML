using CommandLine;
using DotML.Network.IO;

namespace DotML.Cli.Commands;

[Verb("build", HelpText = "Compile a netbuild script into a neural network")]
public class Build : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "File path to model to build", Required = true)]
    public string? FilePath {get; set;}

    [Option("tag", HelpText = "Tag to use to uniquely identify this network once built", Required = false)]
    public string? Tag {get; set;}

    public override void Action(AppData appData) {
        FileInfo file;
        if (string.IsNullOrEmpty(FilePath) || !((file = new FileInfo(FilePath)).Exists)) {
            Console.WriteLine($"The file '{FilePath}' doesn't exist.");
            return;
        }

        NetBuild builder = new NetBuild();
        Console.Write($"Loading build context {DataSize.FromValue(file.Length)}...");
        using var reader = new StreamReader(file.OpenRead());
        var text = reader.ReadToEnd();
        Console.WriteLine("done");
        Console.WriteLine();

        Console.Write($"Parsing build commands...");
        var ast = builder.Parse(text);
        Console.WriteLine("done");
        Console.WriteLine();

        Console.WriteLine($"Building network...");
        var network = ast.Make((index, count, statement) => {
            Console.WriteLine($"Step {index + 1}/{count} : {statement}");
        });
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
        var name = Tag ?? network.Name;

        if (is_valid) {
            var network_file = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + ".netbuild"));
            var meta_file = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + ".xml"));
            File.Copy(file.FullName, network_file.FullName);
            using (var writer = new StreamWriter(meta_file.FullName)) {
                var info = new ModelInfo(meta_file);
                if (name is not null)
                    info.Tags.Add(name);
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