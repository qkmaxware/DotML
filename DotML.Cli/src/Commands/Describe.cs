using System.Security.Cryptography.X509Certificates;
using CommandLine;
using DotML.Network;
using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

namespace DotML.Cli.Commands;

[Verb("describe", HelpText = "Describe the details of an compiled model")]
public class Describe : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    public override void Action(AppData appData) {
        ModelInfo? model = appData.GetModel(ModelName);
        if (model is null)
        {
            WriteError($"No model exists with name '{ModelName}'.");
            return;
        }

        var info_list = new VBox();
        var info_panel = new Panel($"About {model.Guid ?? "Model"}", info_list).WithPadding(1);

        if (!string.IsNullOrEmpty(model.Description))
        {
            info_list.Add(new Label("Description:"));
            info_list.Add(new Paragraph(model.Description).WithMargin(left: 2, bottom: 1, right: 2));
        }

        if (model.ProblemDescription is not null)
        {
            var problem = model.ProblemDescription;
            info_list.Add(new Label("Problem:"));

            if (problem.Classification is not null)
            {
                info_list.Add(new Label($"Type: {nameof(problem.Classification)}"));
                info_list.Add(new Label($"Labels: {string.Join(", ", problem.Classification.ClassLabels)}"));
            }

            
            info_list.Add(new Label(string.Empty));
        }

        if (model.Tags.Any())
        {
            var ul = new UnorderedList();
            foreach (var tag in model.Tags.Select((t, i) => (i, t)))
            {
                ul.Add(new Label(tag.t));
            }
            info_list.Add(new Label("Also known as:"));
            info_list.Add(ul);
        }

        var training_list = new VBox();
        var training_panel = new Panel("Training", training_list).WithPadding(1);
        if (model.Status() == ModelTrainingStatus.Untrained)
        {
            training_list.Add(new Paragraph("This model has not yet been trained."));
        } else
        {
            var trainingMeta = model.TrainingMetadata;
            if (trainingMeta is null)
            {
                training_list.Add(new Paragraph("This model is trained, but no information was recorded about the performance of the model. This model may have been trained outside of this program."));
            }
            else
            {
                string[] train_columns = ["Accuracy ", "Precision", "Recall", "Loss", "Duration"];
                string?[] train_values = [
                    trainingMeta?.Accuracy.ToString(),
                trainingMeta?.Precision.ToString(),
                trainingMeta?.Recall.ToString(),
                $"{trainingMeta?.AvgLoss:F3} ± {(trainingMeta?.MaxLoss - trainingMeta?.MinLoss):F3}",
                trainingMeta?.TrainingDuration?.ToString()
                ];
                var columns = new Columns();
                for (var i = 0; i < train_values.Length; i++)
                {
                    columns.Add(new VBox(
                        new Label(train_columns[i]),
                        new Label(train_values[i])
                    ));
                }
                training_list.Add(columns);
            }
        }

        var script_list = new VBox();
        var script_panel = new Panel("Architecture", script_list).WithPadding(1);
        foreach (var line in model.GetBuildScript().Split('\n'))
        {
            var trimmed = line.TrimEnd();
            if (!string.IsNullOrEmpty(trimmed))
                script_list.Add(new Label(trimmed));
        }

        var app = new RenderView(new VBox(
            info_panel,
            training_panel,
            script_panel
        ));
        app.RenderOnce();
        return;
    }

    private static string TrimEnd(string src, string postfix) {
        if (!src.EndsWith(postfix))
            return src;

        return src.Remove(src.LastIndexOf(postfix));
    }
}