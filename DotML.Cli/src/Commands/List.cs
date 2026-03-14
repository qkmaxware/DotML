using CommandLine;
using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

namespace DotML.Cli.Commands;

// Example output
/*
NETWORK ID  ARCHITECTURE  CREATED       STATUS  
net1        alexnet:v1    Jan 21, 2024  Untrained
net2        alexnet:v1    Jan 21, 2024  Trained
*/

[Verb("list", HelpText = "List all compiled models")]
public class List : BaseCommand {

    [Option("filter", Required = false)]
    public string? Filter {get; set;}

    private class ModelListing
    {
        public string? Name { get; set; }
        public string? Guid { get; set; }
        public string? CreatedDate { get; set; }
        public string? ModifiedDate { get; set; }
        public string? Status { get; set; }
    }

    private class View : ConsoleApp
    {
        public List<ModelListing> models = new List<ModelListing>();
        public View(AppData appData)
        {
            this.Root = new VBox(
                new Panel(
                    "Filesystem",
                    new Label($"Model location: {appData.ModelDirectory}")
                ).WithPadding(1),
                new Table<ModelListing>(models)
            );
        }
    }

    public override void Action(AppData appData)
    {
        var view = new View(appData);

        foreach (var model in appData.ListModels())
        {
            string guid = model.Guid ?? string.Empty;
            IEnumerable<string> tags = model.Tags ?? [];
            foreach (var tag in tags)
            {
                if (
                    tag is null
                    || (
                        !string.IsNullOrEmpty(Filter)
                        && !(
                            tag.Contains(Filter, StringComparison.CurrentCultureIgnoreCase)
                            || guid.Contains(Filter, StringComparison.CurrentCultureIgnoreCase)
                        )
                    )
                )
                {
                    // We were filtering (filter exists) but the name doesn't contain the filter. 
                    // Skip
                    continue;
                }

                view.models.Add(new ModelListing
                {
                    Name = tag,
                    Guid = model.Guid ?? "?",
                    CreatedDate = model.Created().ToString("yyyy-MM-dd hh:mm"),
                    ModifiedDate = model.Modified().ToString("yyyy-MM-dd hh:mm"),
                    Status = model.Status().ToString()
                });
            }
        }

        view.RenderOnce();
    }
}