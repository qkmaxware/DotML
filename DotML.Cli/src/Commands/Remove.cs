using CommandLine;
using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

namespace DotML.Cli.Commands;

// Example output
/*
Untrained network "net1" removed
*/

[Verb("rm", HelpText = "Remove/delete a compiled model")]
public class Remove : BaseCommand {
    
    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    private class View: ConsoleApp
    {
        public string? model_guid = null;
        public string? error = null;

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
                    condition: () => string.IsNullOrEmpty(error),
                    new Panel(
                        "Model Deleted",
                        new Paragraph(() => $"The model {model_guid} has been successfully deleted.")
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

        bool was_deleted = model.Delete();
        if (!was_deleted)
        {
            view.error = $"Unable to remove model '{model.Guid}'. You may try manually deleting the model file here: '{appData.ModelDirectory.FullName}'.";
        }

        view.RenderOnce();
    }
}