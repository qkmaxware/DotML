using CommandLine;
using CommandLine.Text;

namespace DotML.Cli.Commands;

// Example usage
/*
dotml mod net2 tags --add "new-tag" 
dotml mod net2 weights --transfer "my-weights.safetensors"
*/


[Verb("mod", HelpText = "Modify an existing model")]
public class Modify : BaseCommand {

    public enum SubCommand {
        none,
        tags,
        weights
    }

    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    [Value(1, MetaName = "sub-command", HelpText = "Modification type (tags, weights)", Required = true)]
    public SubCommand Cmd {get; set;}

    /*[Usage()]
    public static IEnumerable<Example> Examples {
        get {
            yield return new Example("Add the 'identifier' tag to the model 'my-network'", new Modify { Cmd = SubCommand.tags, ModelName = "my-network", TagsToAdd = ["identifier"] });
            yield return new Example("Delete the 'identifier' tag from the model 'my-network'", new Modify { Cmd = SubCommand.tags, ModelName = "my-network", TagsToRemove = ["identifier"] });
            yield return new Example("Transfer weights from 'weights.safetensors' to the model 'my-network'", new Modify { Cmd = SubCommand.weights, ModelName = "my-network", WeightsToTransfer = "weights.safetensors" });
        }
    }*/

    public override void Action(AppData appData) {
        switch (Cmd) {
            case SubCommand.tags: TagAction(appData); break;
            case SubCommand.weights: WeightsAction(appData); break;
        }
    }

    #region Tagging
    [Option("add", Required = false, HelpText = "In 'tagging' mode, tags to add to the model", Separator = ' ')]
    public IEnumerable<string>? TagsToAdd {get; set;}

    [Option("rm", Required = false, HelpText = "In 'tagging' mode, tags to remove from the model", Separator = ' ')]
    public IEnumerable<string>? TagsToRemove {get; set;}

    public void TagAction(AppData appData) {
        var model = appData.GetModel(ModelName);
        if (model is null) {
            Console.WriteLine($"No model exists with name '{ModelName}'.");
            return;
        }

        if (TagsToAdd is not null && TagsToAdd.Any()) {
            foreach (var tag in TagsToAdd)
                model.Tags.Add(tag);
            Console.WriteLine($"Successfully added tags [{string.Join(',', TagsToAdd)}] on {model.Guid}");
        }

        if (TagsToRemove is not null && TagsToRemove.Any()) {
            foreach (var tag in TagsToRemove)
                model.Tags.Remove(tag);
            Console.WriteLine($"Successfully removed tags [{string.Join(',', TagsToRemove)}] on {model.Guid}");
        }

        model.UpdateMetadata();
    }
    #endregion

    #region Weight-transfer
    [Option("import", Required = false, HelpText = "In 'weight' mode, import the given weights to the model", Separator = ' ')]
    public string? WeightsToImport {get; set;}

    [Option("export", Required = false, HelpText = "In 'weight' mode, export the given weights to the model", Separator = ' ')]
    public string? WeightsToExport {get; set;}

    [Option("transfer", Required = false, HelpText = "In 'weight' mode, transfer weights to another model", Separator = ' ')]
    public string? WeightsToTransfer {get; set;}

    public void WeightsAction(AppData appData) {
        var model = appData.GetModel(ModelName);
        if (model is null) {
            Console.WriteLine($"No model exists with name '{ModelName}'.");
            return;
        }
        
        if (!string.IsNullOrEmpty(WeightsToImport)) {
            var weight_file = new FileInfo(WeightsToImport);
            if (weight_file.Exists) {
                model.UpdateWeights(weight_file);
                Console.WriteLine($"Successfully imported weights '{WeightsToImport}' to {model.Guid}");
            }
        }

        if (!string.IsNullOrEmpty(WeightsToExport) && model.Status() == ModelTrainingStatus.Trained) {
            var weights = model.GetWeightsFile();

            if (weights is not null && weights.Exists) {
                using var ostream = new FileStream(WeightsToExport, FileMode.Create);
                using var istream = weights.OpenRead();
                istream.CopyTo(ostream);

                Console.WriteLine($"Successfully exported weights from {model.Guid} to '{WeightsToExport}'");
            }
        }

        if (!string.IsNullOrEmpty(WeightsToTransfer)) {
            var model_to = appData.GetModel(WeightsToTransfer);
            var weights_from = model.GetWeightsFile();
            if (weights_from is not null && weights_from.Exists && model_to is not null) {
                model_to.UpdateWeights(weights_from);
                Console.WriteLine($"Successfully transferred weights from {model.Guid} to {model_to.Guid}");
            }   
        }
    }
    #endregion

}