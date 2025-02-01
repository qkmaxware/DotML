using CommandLine;

namespace DotML.Cli.Commands;

// Example output
/*
Untrained network "net1" removed
*/

[Verb("rm", HelpText = "Remove/delete a compiled network")]
public class Remove : BaseCommand {
    
    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    public override void Action(AppData appData) {
        var model = appData.GetModel(ModelName);
        if (model is null) {
            Console.WriteLine($"No model exists with name '{ModelName}'.");
            return;
        }

        bool was_deleted = model.Delete();
        if (was_deleted) {
            Console.WriteLine($"{model.Status()} model '{model.Guid}' removed.");
        } else {
            Console.WriteLine($"Unable to remove model '{model.Guid}'. You may try manually deleting the model file here: '{appData.ModelDirectory.FullName}'.");
        }
    }
}