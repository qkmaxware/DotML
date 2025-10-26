using System.Diagnostics;
using CommandLine;
using DotML.Network;
using DotML.Network.Training;
using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

namespace DotML.Cli.Commands;

[Verb("test", HelpText = "Test a model against a dataset")]
public class Test : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    [Option("data-test", HelpText = "The data-set used for testing", Required = false)]
    public string? TestingDataPath {get; set;}

    public override void Action(AppData appData)
    {
        // Construct the view
        var view_stack = new VBox();
        var view = new RenderView(view_stack);

        string? error_string = null;
        var error_message = new Paragraph(() => error_string ?? string.Empty);
        var error_option_list = new List<IElement>();
        var error_panel = new Panel("Error", new VBox(
            error_message,
            new Conditional(() => error_option_list.Count > 0,
                new VBox(
                    new Label("Options:"),
                    new UnorderedList(error_option_list)
                )
            )
        )).WithPadding(1);
        view_stack.Add(new Conditional(() => !string.IsNullOrEmpty(error_string), error_panel));

        var success_view = new VBox();
        view_stack.Add(new Conditional(() => string.IsNullOrEmpty(error_string), success_view));

        var process_list = new UnorderedList();
        var process_panel = new Panel("Process", process_list).WithPadding(1);
        success_view.Add(process_panel);

        var loadModelTaskState = TaskState.Waiting;
        var loadModelTaskUi = MakeTaskView(() => loadModelTaskState, "Load Model");
        process_list.Add(loadModelTaskUi);

        var loadDataTaskState = TaskState.Waiting;
        var loadDataTaskUi = MakeTaskView(() => loadDataTaskState, "Load testing data");
        process_list.Add(loadDataTaskUi);

        var testingTaskState = TaskState.Waiting;
        float testingProgress = 0.0f;
        var testingTaskUi = MakeTaskView(() => testingTaskState, () => testingProgress, "Testing");
        process_list.Add(testingTaskUi);

        var output_content_area = new Qkmaxware.Terminal.Layout.Padding(null, 0, 0, 0, 0);
        success_view.Add(new Conditional(() => output_content_area.ChildComponent is not null, output_content_area));

        // Start the actual task
        var task = Task.Run(() => DoTasks(appData, ref error_string, ref loadModelTaskState, ref loadDataTaskState, ref testingTaskState, ref testingProgress, ref output_content_area));

        // Wait (while rendering progress)
        var (left, top) = Console.GetCursorPosition();
        view.RenderWhile((self) => !task.IsCompleted);

        // Final render 
        if (task.IsFaulted)
        {
            error_string = task.Exception.Message;
            error_option_list.Clear();
        }
        Console.SetCursorPosition(left, top);
        view.RenderOnce();
    }

    private struct testingResult
    {
        public float Accuracy { get; set; }
        public float Precision { get; set; }
        public float Recall { get; set; }
        public float AvgLoss { get; set; }
        public string Validation { get; set; }
        public TimeSpan TimeTaken { get; set; }
    }

    private void DoTasks(AppData appData, ref string? error_string, ref TaskState loadModel, ref TaskState loadData, ref TaskState training, ref float trainingProgress, ref Qkmaxware.Terminal.Layout.Padding resultArea)
    {
        var model = appData.GetModel(ModelName);
        if (model is null)
        {
            error_string = ($"No model exists with name '{ModelName}'.");
            return;
        }
        if (string.IsNullOrEmpty(TestingDataPath) || !IsPathValid(TestingDataPath))
        {
            Console.WriteLine($"The testing data path '{TestingDataPath}' is invalid or the file doesn't exist.");
            return;
        }

        loadModel = TaskState.Running;
        var network = model.Load();
        loadModel = TaskState.Done;

        loadData = TaskState.Running;
        var data = Fit.ReadTrainingData(TestingDataPath);
        loadData = TaskState.Done;

        loadData = TaskState.Running;
        LossFunction loss = Fit.SmartPickLoss(model);
        var watch = Stopwatch.StartNew();
        var (console_left, console_top) = Console.GetCursorPosition();
        var report = ModuleTrainingEnumerator.Test(
            data.CreateSequentialSampler(),
            network,
            loss,
            Environment.ProcessorCount
        );
        watch.Stop();
        var elapsed = watch.Elapsed;
        Console.SetCursorPosition(console_left, console_top);
        loadData = TaskState.Done;

        // Draw console report
        List<testingResult> tableRows = new List<testingResult>();
        var table = new Table<testingResult>(tableRows);

        tableRows.Add(new testingResult
        {
            Accuracy = (float)report.Accuracy,
            Precision = (float)report.Precision,
            Recall = (float)report.Recall,
            AvgLoss = (float)report.AvgLoss,
            Validation = report.TestsPassedCount + "/" + report.SampleCount,
            TimeTaken = elapsed
        });

        // Write reports
        var report_dir = appData.CreateTestingDir();
        using (var writer = new StreamWriter(Path.Combine(report_dir.FullName, $"command.sh")))
        {
            writer.WriteLine(Environment.CommandLine);
        }
        using (var writer = new StreamWriter(Path.Combine(report_dir.FullName, $"model.{model.Guid}.xml")))
        {
            writer.Write(model.ToXml());
        }
        using (var writer = new StreamWriter(Path.Combine(report_dir.FullName, $"summary.csv")))
        {
            writer.WriteLine("LOSS-AVERAGE, LOSS-MIN, LOSS-MAX, ACCURACY, PRECISION, RECALL, F1-SCORE, TIME-TAKEN");
            writer.Write(report.AvgLoss); writer.Write(',');
            writer.Write(report.MinLoss); writer.Write(',');
            writer.Write(report.MaxLoss); writer.Write(',');
            writer.Write(report.Accuracy); writer.Write(',');
            writer.Write(report.Precision); writer.Write(',');
            writer.Write(report.Recall); writer.Write(',');
            writer.Write(report.F1); writer.Write(',');
            writer.Write($" \"{elapsed}\""); writer.WriteLine();
        }
        if (TryGetEmbeddedDoc("src/Docs/TestingReports.md", out string? documentation))
        {
            using (var writer = new StreamWriter(Path.Combine(report_dir.FullName, $"readme.md")))
            {
                writer.Write(documentation);
            }
        }

        resultArea.ChildComponent = new VBox(
            table,
            new Label($"Reports saved to '{report_dir.FullName}'."),
            new Label($"Use \"{typeof(Fit).Assembly.GetName().Name} reports open '{report_dir.Name}'\" to review testing metrics.")
        );
    }
}