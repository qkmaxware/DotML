using System.IO.Compression;
using CommandLine;
using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

namespace DotML.Cli.Commands;

[Verb("export", HelpText = "Export a model to a single file for sharing")]
public class Export : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    [Option('o', "out", HelpText = "Output file path", Required = false)]
    public string? OutPath {get; set;}

    private class View : ConsoleApp
    {
        public string? model_guid = null;
        public string? error = null;
        public string? architecture_path = null;
        public string? metadata_path = null;
        public string? weights_path = null;
        public string? output_path = null; 

        public View()
        {
            this.Root = new VBox(
                new Conditional(
                    condition: () => !string.IsNullOrEmpty(error),
                    new Panel(
                        "Error",
                        new Paragraph(() => this.error ?? "An unexpected error has occurred")
                    ).WithPadding(1)
                ),
                new Conditional(
                    condition: () => !string.IsNullOrEmpty(architecture_path),
                    new Panel(
                        "Model Exported",
                        new VBox(
                            new Paragraph(() => $"Model {model_guid} exported to '{output_path}'."),
                            new Label("Files:").WithMargin(top: 1),
                            new UnorderedList(
                                new Conditional(
                                    () => !string.IsNullOrEmpty(metadata_path),
                                    new Label(() => $"Model metadata ({metadata_path})")
                                ),
                                new Conditional(
                                    () => !string.IsNullOrEmpty(architecture_path),
                                    new Label(() => $"Architecture description ({architecture_path})")
                                ),
                                new Conditional(
                                    () => !string.IsNullOrEmpty(weights_path),
                                    new Label(() => $"Trained weights ({weights_path})")
                                )
                            )
                        )
                    ).WithPadding(1)                  
                )
            );
        }
    }

    public override void Action(AppData appData)
    {
        var view = new View();
        var model = appData.GetModel(ModelName);
        if (model is null)
        {
            view.error = $"No model exists with name '{ModelName}'.";
            view.RenderOnce();
            return;
        }
        view.model_guid = model.Guid;

        var def_name = (model.Tags?.FirstOrDefault() ?? model.Guid ?? "netflow") + ".model";
        var result = OutPath ?? def_name;
        if (!result.EndsWith(".zip"))
        {
            result += ".zip";
        }

        if (File.Exists(result))
            File.Delete(result);

        using (ZipArchive archive = ZipFile.Open(result, ZipArchiveMode.Create))
        {
            var build_path = model.GetBuildScriptFile();
            if (build_path is not null && build_path.Exists)
            {
                var entry = archive.CreateEntryFromFile(
                    build_path.FullName,
                    "architecture" + build_path.Extension
                );
                view.architecture_path = "architecture" + build_path.Extension;
            }

            var meta_path = model.GetMetadataFile();
            if (meta_path is not null && meta_path.Exists)
            {
                var entry = archive.CreateEntryFromFile(
                    meta_path.FullName,
                    "metadata.xml"
                );
                view.metadata_path = "metadata.xml";
            }

            var weight_path = model.GetWeightsFile();
            if (weight_path is not null && weight_path.Exists)
            {
                var entry = archive.CreateEntryFromFile(
                    weight_path.FullName,
                    "weights.safetensors"
                );
                view.weights_path = "weights.safetensors";
            }
        }

        view.output_path = result;
        view.RenderOnce();
    }
}

[Verb("import", HelpText = "Import a model shared from another machine")]
public class Import : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "File path to zipped model", Required = true)]
    public string? FilePath {get; set;}

    private class View : ConsoleApp
    {
        public string? model_guid = null;
        public string? error = null;
        public string? architecture_path = null;
        public string? metadata_path = null;
        public string? weights_path = null;
        public string? input_path = null; 

        public View()
        {
            this.Root = new VBox(
                new Conditional(
                    condition: () => !string.IsNullOrEmpty(error),
                    new Panel(
                        "Error",
                        new Paragraph(() => this.error ?? "An unexpected error has occurred")
                    ).WithPadding(1)
                ),
                new Conditional(
                    condition: () => !string.IsNullOrEmpty(architecture_path),
                    new Panel(
                        "Model Imported",
                        new VBox(
                            new Paragraph(() => $"Successfully imported '{input_path}' as model {model_guid}."),
                            new Label("Files:").WithMargin(top: 1),
                            new UnorderedList(
                                new Conditional(
                                    () => !string.IsNullOrEmpty(metadata_path),
                                    new Label(() => $"Model metadata ({metadata_path})")
                                ),
                                new Conditional(
                                    () => !string.IsNullOrEmpty(architecture_path),
                                    new Label(() => $"Architecture description ({architecture_path})")
                                ),
                                new Conditional(
                                    () => !string.IsNullOrEmpty(weights_path),
                                    new Label(() => $"Trained weights ({weights_path})")
                                )
                            )
                        )
                    ).WithPadding(1)                  
                )
            );
        }
    }

    public override void Action(AppData appData)
    {
        var view = new View();
        FileInfo file;
        if (string.IsNullOrEmpty(FilePath) || !((file = new FileInfo(FilePath)).Exists))
        {
            view.error = $"The file '{FilePath}' doesn't exist.";
            view.RenderOnce();
            return;
        }

        var guid = Guid.NewGuid().ToString();
        view.model_guid = guid;
        view.input_path = FilePath;

        bool has_build = false; bool has_meta = false;
        using (ZipArchive archive = ZipFile.Open(FilePath, ZipArchiveMode.Read))
        {
            FileInfo? network_file = null;
            var build_entry = archive.Entries.Where((entry) => entry.Name.StartsWith("architecture")).FirstOrDefault();
            if (build_entry is not null)
            {
                network_file = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + Path.GetExtension(build_entry.Name)));
                build_entry.ExtractToFile(network_file.FullName);
                has_build = true;
                view.architecture_path = build_entry.Name;
            }

            var meta_file = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + ".xml"));
            var meta_entry = archive.GetEntry("metadata.xml");
            if (meta_entry is not null)
            {
                meta_entry.ExtractToFile(meta_file.FullName);
                has_meta = true;
                view.metadata_path = "metadata.xml";
            }

            var weights_file = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + ".safetensors"));
            var weights_entry = archive.GetEntry("weights.safetensors");
            if (weights_entry is not null)
            {
                weights_entry.ExtractToFile(weights_file.FullName);
                view.weights_path = "weights.safetensors";
            }
        }


        if (!(has_build && has_meta))
        {
            view.error = $"Failed to import model from '{FilePath}'.";
        }
        view.RenderOnce();
    }
}