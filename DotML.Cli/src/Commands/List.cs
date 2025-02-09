using CommandLine;

namespace DotML.Cli.Commands;

// Example output
/*
NETWORK ID  ARCHITECTURE  CREATED       STATUS  
net1        alexnet:v1    Jan 21, 2024  Untrained
net2        alexnet:v1    Jan 21, 2024  Trained
*/

[Verb("list", HelpText = "List all compiled networks")]
public class List : BaseCommand {

    [Option("filter", Required = false)]
    public string? Filter {get; set;}

    public override void Action(AppData appData) {
        string[] columns = ["NAME/TAG", "ARCHITECTURE", "CREATED", "MODIFIED", "STATUS"];
        int[] column_lengths = [36, 36, 16, 16, 24];

        for (var col = 0; col < columns.Length; col++) {
            var name = columns[col];
            var len = column_lengths[col];
            Console.Write(ColumnValue(name, len));
            Console.Write(' ');
        }
        Console.WriteLine();

        foreach (var model in appData.ListModels()) {
            IEnumerable<string> tags = model.Tags;
            if (model.Tags is null || model.Tags.Count < 1) {
                model.Tags = [model.Guid];
            }
            foreach (var tag in tags) {
                if (tag is null || (!string.IsNullOrEmpty(Filter) && !tag.Contains(Filter, StringComparison.CurrentCultureIgnoreCase))) {
                    // We were filtering (filter exists) but the name doesn't contain the filter. 
                    // Skip
                    continue;
                }

                Console.Write(ColumnValue(tag, column_lengths[0]));
                Console.Write(' ');

                Console.Write(ColumnValue(model.Guid ?? "?", column_lengths[1]));
                Console.Write(' ');

                Console.Write(ColumnValue(model.Created().ToString("yyyy-MM-dd hh:mm"), column_lengths[2]));
                Console.Write(' ');

                Console.Write(ColumnValue(model.Modified().ToString("yyyy-MM-dd hh:mm"), column_lengths[3]));
                Console.Write(' ');

                Console.Write(ColumnValue(model.Status(), column_lengths[4]));
                Console.WriteLine();
            }
        }   
    }
}