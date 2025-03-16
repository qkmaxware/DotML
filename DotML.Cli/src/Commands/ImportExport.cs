using System.IO.Compression;
using CommandLine;

namespace DotML.Cli.Commands;

[Verb("export", HelpText = "Export a model to a single file for sharing")]
public class Export : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    [Option('o', "out", HelpText = "Output file path", Required = false)]
    public string? OutPath {get; set;}

    public override void Action(AppData appData) {
        var model = appData.GetModel(ModelName);
        if (model is null) {
            Console.WriteLine($"No model exists with name '{ModelName}'.");
            return;
        }

        var def_name = (model.Tags?.FirstOrDefault() ?? model.Guid ?? "model");
        var result = OutPath ?? def_name;
        if (!result.EndsWith(".zip")) {
            result += ".zip";
        }

        using (ZipArchive archive = ZipFile.Open(result, ZipArchiveMode.Create)) {
            var build_path = model.GetBuildScriptFile();
            if (build_path is not null && build_path.Exists) {
                var entry = archive.CreateEntryFromFile(
                    build_path.FullName, 
                    "architecture.netbuild"
                );
            }

            var meta_path = model.GetMetadataFile();
            if (meta_path is not null && meta_path.Exists) {
                var entry = archive.CreateEntryFromFile(
                    meta_path.FullName,
                    "metadata.xml"
                );
            }

            var weight_path = model.GetWeightsFile();
            if (weight_path is not null && weight_path.Exists) {
                var entry = archive.CreateEntryFromFile(
                    weight_path.FullName,
                    "weights.safetensors"
                );
            }
        }

        Console.WriteLine($"Model {model.Guid} exported to 'result'.");
    }
}

[Verb("import", HelpText = "Import a model shared from another machine")]
public class Import : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "File path to zipped model", Required = true)]
    public string? FilePath {get; set;}

    public override void Action(AppData appData) {
        FileInfo file;
        if (string.IsNullOrEmpty(FilePath) || !((file = new FileInfo(FilePath)).Exists)) {
            Console.WriteLine($"The file '{FilePath}' doesn't exist.");
            return;
        }

        var guid = Guid.NewGuid().ToString();
        bool has_build = false; bool has_meta = false;
        using (ZipArchive archive = ZipFile.Open(FilePath, ZipArchiveMode.Create)) {
            var network_file = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + ".netbuild"));
            var build_entry = archive.GetEntry("architecture.netbuild");
            if (build_entry is not null) {
                build_entry.ExtractToFile(network_file.FullName);
                has_build = true;
            }

            var meta_file = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + ".xml"));
            var meta_entry = archive.GetEntry("metadata.xml");
            if (meta_entry is not null) {
                meta_entry.ExtractToFile(meta_file.FullName);
                has_meta = true;
            }

            var weights_file = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + ".safetensors"));
            var weights_entry = archive.GetEntry("weights.safetensors");
            if (weights_entry is not null) {
                weights_entry.ExtractToFile(weights_entry.FullName);
            }
        }

        if (has_build && has_meta) {
            Console.WriteLine($"Successfully imported model {guid}.");
        } else {
            Console.WriteLine($"Failed to import model from '{FilePath}'.");
        }
    }
}