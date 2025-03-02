using System.Drawing;
using System.Numerics;
using CommandLine;
using DotML.Cli.Visualizations;

namespace DotML.Cli.Commands;

[Verb("report", HelpText = "View a report summarizing the progress of a particular training session.")]
public class Report : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "Report name", Required = false)]
    public string? ReportName {get; set;}

    public override void Action(AppData appData) {
        if (ReportName is null) {
            Console.WriteLine("No report name provided");
            Console.WriteLine();

            Console.WriteLine("Available reports:");
            foreach (var session in appData.EnumerateTrainingSessions()) {
                Console.Write("- "); Console.WriteLine(session.Name);
            }
            return;
        }

        var report = appData.EnumerateTrainingSessions().Where(dir => dir.Name.Contains(ReportName, StringComparison.CurrentCultureIgnoreCase)).FirstOrDefault();
        if (report is null) {
            Console.WriteLine($"No training reports exist with name '{ReportName}'.");
            return;
        }

        var model = report.ModelInfo;
        if (model is not null) {
            Console.WriteLine("MODEL");
            Console.WriteLine($"Tags: {string.Join(',', model.Tags)}");
            Console.WriteLine();
        }

        var trainer_params = report.TrainerConfig;
        if (trainer_params is not null) {
            Console.WriteLine("TRAINER");
            Console.WriteLine(File.ReadAllText(trainer_params.FullName));
            Console.WriteLine(); 
        }

        var weights = report.RetainedWeights;
        if (weights is not null && weights.Any()) {
            Console.WriteLine("WEIGHTS");
            foreach (var weight in weights.OrderBy(x => x.Name, new AlphaNumericComparer())) {
                Console.Write("- ");
                Console.WriteLine(weight.Name);
            }
            Console.WriteLine(); 
        }

        var testing_data = report.TestingChart;
        var validation_data = report.ValidationChart;
        var final_data = testing_data ?? validation_data;

        if (final_data is not null) {
            var data = ReadCsv(final_data);
            var loss = data.Where(row => row.ContainsKey("Loss-Average")).Select((row, i) => new Vector2(i, float.Parse(row["Loss-Average"])));

            Console.WriteLine("LOSS");
            Plot2D plot = new Plot2D(0..10, 0..2);
            plot.Draw(loss);
        }
    }

    private static List<Dictionary<string, string>> ReadCsv(FileInfo file) {
        var headers = new List<string>();
        List<Dictionary<string, string>> items = new List<Dictionary<string, string>>();

        bool first_row = true;
        using var reader = new StreamReader(file.OpenRead());
        string? line;
        while ((line = reader.ReadLine()) is not null) {
            var data = line.Split(',');

            if (first_row) {
                // Skip the header
                headers = data.ToList();
                first_row = false;
            }

            Dictionary<string, string> row = new Dictionary<string, string>();
            for (var i = 0; i < data.Length; i++) {
                if (i >= 0 && i < headers.Count) {
                    row[headers[i]] = data[i];
                } else {
                    row[i.ToString()] = data[i];
                }
            }
            items.Add(row);

            first_row = false;
        }
        
        return items;
    }
}