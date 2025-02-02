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

[Verb("tag", HelpText = "Tag an model with a new tag or remove an existing tag")]
public class Tag : BaseCommand {
    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    [Option('a', "add", Required = false, HelpText = "Tags to add to the model", Separator = ' ')]
    public IEnumerable<string>? ToAdd {get; set;}

    [Option('r', "rm", Required = false, HelpText = "Tags to remove from the model", Separator = ' ')]
    public IEnumerable<string>? ToRemove {get; set;}

    public override void Action(AppData appData) {
        var model = appData.GetModel(ModelName);
        if (model is null) {
            Console.WriteLine($"No model exists with name '{ModelName}'.");
            return;
        }

        if (ToAdd is not null && ToAdd.Any()) {
            foreach (var tag in ToAdd)
                model.Tags.Add(tag);
            Console.WriteLine($"Successfully added tags [{string.Join(',', ToAdd)}] on {model.Guid}");
        }

        if (ToRemove is not null && ToRemove.Any()) {
            foreach (var tag in ToRemove)
                model.Tags.Remove(tag);
            Console.WriteLine($"Successfully removed tags [{string.Join(',', ToRemove)}] on {model.Guid}");
        }

        model.UpdateMetadata();
    }
}