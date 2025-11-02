using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using CommandLine;
using DotML.Cli.Notifiers;
using DotML.Cli.Retention;
using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.Training;
using DotML.Cli.Visualizations;
using Qkmaxware.Terminal.Layout;
using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using DotML.Network.IO;

namespace DotML.Cli.Commands;

[Verb("fit", HelpText = "Train a compiled model against a provided training data set")]
public class Fit : BaseCommand
{
    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName { get; set; }

    [Option("data-training", HelpText = "The data-set used for training", Required = true)]
    public string? TrainingDataPath { get; set; }

    [Option("data-validation", HelpText = "The data-set used for validation", Required = false)]
    public string? ValidationDataPath { get; set; }

    [Option("data-test", HelpText = "The data-set used for testing", Required = false)]
    public string? TestingDataPath { get; set; }

    [Option("epochs", HelpText = "Maximum number of epochs to train for (min 1)", Required = false, Default = 100)]
    public int MaxEpochs { get; set; }

    [Option("learning-rate", HelpText = "Initial learning rate", Required = false, Default = 0.01)]
    public double LearningRate { get; set; }

    [Option("accuracy", HelpText = "Early stop accuracy threshold", Required = false, Default = 0.15)]
    public double Accuracy { get; set; }

    [Option("patience", HelpText = "Early stop patience (min 1)", Required = false, Default = 1)]
    public int Patience { get; set; }

    [Option("batch-size", HelpText = "Batch size, leave blank for batch to be automatically determined", Required = false)]
    public int? BatchSize { get; set; }

    [Option("loss", HelpText = "Name of the loss function to use, leave blank for function to be automatically determined", Required = false)]
    public string? LossFunctionName { get; set; }

    [Option("init", HelpText = "Name of the initialization method to use, leave blank for function to be automatically determined", Required = false)]
    public string? InitializerName { get; set; }

    // TODO !!!!!
    [Option("clip-gradients", HelpText = "Flag to indicate if gradient clipping should be performed", Required = false, Default = "false")]
    public string? ClipGradientsStr { get; set; }
    public bool ClipGradients => IsSet(ClipGradientsStr);

    [Option("continue", HelpText = "Flag to indicate if the existing weights should be used or if the network should be re-initialized", Required = false, Default = "false")]
    public string? ContinueStr { get; set; }
    public bool ContinueFromExisting => IsSet(ContinueStr);

    [Option("diagram", HelpText = "Flag to indicate if the network should be rendered as a diagram", Required = false, Default = "false")]
    public string? DiagramStr { get; set; }

    public enum UpdateModeType
    {
        none, overwrite, duplicate
    }
    [Option("save", HelpText = "Flag to indicate how the final weights should be applied to the model (none, overwrite, or duplicate)", Default = UpdateModeType.overwrite)]
    public UpdateModeType UpdateMode { get; set; }

    [Option("retention", HelpText = "Flag to indicate how intermediate weights should be retained (none, all, most_recent, last5, last10, smallest_loss, highest_accuracy, most_passed)", Default = null)]
    public IEnumerable<string>? Retention { get; set; }


    [Option("notify", HelpText = "Url, endpoint, or address to send notification updates to.", Required = false, Default = null)]
    public string? NotifyEndpoint { get; set; }

    [Option("notify-on", HelpText = "A list of events which trigger notifications", Required = false, Default = new NotificationEvent[] { NotificationEvent.all })]
    public IEnumerable<NotificationEvent>? NotificationEvents { get; set; }
    private int NotificationTrigger
    {
        get
        {
            int trigger = 0;
            if (NotificationEvents is not null)
            {
                foreach (var evt in NotificationEvents)
                {
                    trigger |= (int)evt;
                }
            }
            return trigger;
        }
    }

    public override void Action(AppData appData)
    {
        #region Validate Args
        var model = appData.GetModel(ModelName);
        if (model is null)
        {
            Console.WriteLine($"No model exists with name '{ModelName}'.");
            return;
        }
        if (string.IsNullOrEmpty(TrainingDataPath) || !IsPathValid(TrainingDataPath))
        {
            Console.WriteLine($"The training data path '{TrainingDataPath}' is invalid or the file doesn't exist.");
            return;
        }
        if (!string.IsNullOrEmpty(ValidationDataPath) && !IsPathValid(ValidationDataPath))
        {
            Console.WriteLine($"The validation data path '{TrainingDataPath}' is invalid or the file doesn't exist.");
            return;
        }
        if (!string.IsNullOrEmpty(TestingDataPath) && !IsPathValid(TestingDataPath))
        {
            Console.WriteLine($"The testing data path '{TestingDataPath}' is invalid or the file doesn't exist.");
            return;
        }
        INotifier? notifier = GetNotifierFor(NotifyEndpoint);
        #endregion

        #region Directory Initialization
        var dir = appData.CreateTrainingDir();
        var weights_dir = Path.Combine(dir.FullName, "weights");
        Directory.CreateDirectory(weights_dir);
        using (var writer = new StreamWriter(Path.Combine(dir.FullName, $"model.{model.Guid}.xml")))
        {
            writer.Write(model.ToXml());
        }
        using (var writer = new StreamWriter(Path.Combine(dir.FullName, $"command.sh")))
        {
            writer.WriteLine(Environment.CommandLine);
        }
        if (TryGetEmbeddedDoc("src/Docs/TrainingReports.md", out string? documentation))
        {
            using (var writer = new StreamWriter(Path.Combine(dir.FullName, $"readme.md")))
            {
                writer.Write(documentation);
            }
        }
        #endregion

        #region Network Load
        var network = model.Load();
        if (network is IMarkdownable md)
        {
            using (var writer = new StreamWriter(Path.Combine(dir.FullName, "network.md")))
            {
                md.ToMarkdown(writer);
            }
        }
        else if (network is IJsonizable json)
        {
            using (var writer = new StreamWriter(Path.Combine(dir.FullName, "network.json")))
            {
                json.ToJson(writer);
            }
        }
        else if (network is IHtmlable html)
        {
            using (var writer = new StreamWriter(Path.Combine(dir.FullName, "network.html")))
            {
                html.ToHtml(writer);
            }
        }

        if (IsSet(DiagramStr))
        {
            if (network is IDiagrammable svg)
            {
                using (var writer = new StreamWriter(Path.Combine(dir.FullName, "network.svg")))
                {
                    svg.ToSvg(writer);
                }
            }
            else if (network is IBlockVisitable visitable)
            {
                var renderer = new SvgRenderer();
                using (var writer = new StreamWriter(Path.Combine(dir.FullName, "network.svg")))
                {
                    renderer.RenderToStream(network, writer);
                }
            }
        }
        #endregion

        #region Trainer Config
        var trainer = new ModuleTrainer();
        trainer.BatchSize = AutoPickBatchSize();
        trainer.MaxEpochs = Math.Max(1, MaxEpochs);
        trainer.LearningRate = MathF.Max(0, (float)LearningRate);
        trainer.LearningRateScheduler = null;
        trainer.Optimizer = new Adam();
        trainer.StopCondition = (report) => report.Loss.Max < Accuracy;
        trainer.Patience = Math.Max(1, Patience);
        trainer.Regularization = new NoRegularization();
        trainer.LocalClipping = null;
        trainer.GlobalClipping = null;
        trainer.Initializer = AutoPickInitializer(model, network);
        trainer.Loss = AutoPickLoss(model, network);

        // Dump settings to file
        using (var trainer_prop_writer = new StreamWriter(Path.Combine(dir.FullName, "trainer-config.yaml")))
        {
            trainer_prop_writer.WriteLine("Trainer:");
            trainer_prop_writer.Write("    "); trainer_prop_writer.Write("Type"); trainer_prop_writer.Write(": "); trainer_prop_writer.WriteLine(trainer.GetType().Name);
            foreach (PropertyInfo property in trainer.GetType().GetProperties())
            {
                object? value = property.CanRead ? property.GetValue(trainer, null) : null;
                if (value is LossFunction loss)
                    value = loss.Name;
                else
                    value = value?.ToString() ?? "n/a";
                trainer_prop_writer.Write("    "); trainer_prop_writer.Write(property.Name); trainer_prop_writer.Write(": "); trainer_prop_writer.WriteLine(value);
            }
        }
        #endregion

        #region Training Data
        var trainingData = ReadTrainingData(TrainingDataPath);
        var validationData = !string.IsNullOrEmpty(ValidationDataPath) ? ReadTrainingData(ValidationDataPath) : Subsample(trainingData, 0.15f, 10);
        var TestingData = !string.IsNullOrEmpty(TestingDataPath) ? ReadTrainingData(TestingDataPath) : null;
        #endregion

        #region Header Display
        {
            var viewStack = new VBox();
            var view = new RenderView(viewStack);

            // Row 0 Network/Data
            {
                var row = new Columns();
                viewStack.Add(row);
                // Network info
                {
                    var items = new VBox();
                    var panel = new Panel("Network", items);
                    row.Add(items);

                    items.Add(LabeledText("Id", model.Guid ?? "?"));
                    items.Add(LabeledText("Name", network.Name()));
                    items.Add(LabeledText("Description", model.Description ?? string.Empty));
                }
                // Training data info
                {
                    var items = new VBox();
                    var panel = new Panel("Data", items);
                    row.Add(items);

                    items.Add(LabeledText("Input Shape", trainingData.InputShape.ToString()));
                    items.Add(LabeledText("Output Shape", trainingData.OutputShape.ToString()));
                    items.Add(LabeledText("Training Pairs", trainingData.Count.ToString()));
                    items.Add(LabeledText("Validation Pairs", validationData.Count.ToString()));
                    items.Add(LabeledText("Testing Pairs", (TestingData?.Count ?? 0).ToString()));
                }
            }

            // Row 1 Trainer
            {
                var grid = new GridBox(3);
                var panel = new Panel("Trainer", grid);
                viewStack.Add(panel);

                foreach (PropertyInfo property in trainer.GetType().GetProperties())
                {
                    object? value = property.CanRead ? property.GetValue(trainer, null) : null;
                    string str = value switch
                    {
                        LossFunction loss => loss.Name,
                        _ => value?.ToString() ?? "n/a"
                    };
                    grid.Add(LabeledText(property.Name, str));
                }
            }

            view.RenderOnce();
            Console.WriteLine();
        }
        #endregion

        #region Training Loop
        var session = (ModuleTrainingEnumerator)trainer.EnumerateTraining(network, trainingData.CreateRandomSampler(), validationData.CreateSequentialSampler());
        session.Reset();

        List<IRetentionPolicy<Safetensors>> retention_policies = this.Retention?.Select(policy => {
            IRetentionPolicy<Safetensors> retention_policy = policy.Trim() switch {
                "all"             => new AllWeights(weights_dir),
                "most_recent"     => new MostRecentWeights(weights_dir),
                string s when s.StartsWith("last") => new LastNWeights(weights_dir, int.Parse(string.Concat(s.Where( Char.IsDigit )))),
                "smallest_loss"   => new SmallestLoss(weights_dir, session.Current),
                "highest_accuracy"=> new HighestAccuracy(weights_dir, session.Current),
                "most_passed"     => new MostTestsPassed(weights_dir, session.Current),
                _                 => new NoWeights()
            };
            return retention_policy;
        })?.ToList() ?? new();

        if (ContinueFromExisting)
        {
            model.ReloadWeights(network);
        }

        using var validation_writer = new StreamWriter(Path.Combine(dir.FullName, "validation.csv"));
        validation_writer.WriteLine("Epoch, Tests-Passed, Tests-Failed, Loss-Average, Loss-Max, Loss-Min, Accuracy, Precision, Recall, F1-Score, Time-Taken");
        validation_writer.Flush();
    
        using var testing_writer = new StreamWriter(Path.Combine(dir.FullName, "testing.csv"));
        testing_writer.WriteLine("Epoch, Tests-Passed, Tests-Failed, Loss-Average, Loss-Max, Loss-Min, Accuracy, Precision, Recall, F1-Score, Time-Taken");
        testing_writer.Flush();

        if (IsSet(NotificationEvent.started))
            notifier?.NotifyTrainingStarted(network);

        if (TestingData is not null)
        {
            var testTimer = Stopwatch.StartNew();
            var report = session.Test(TestingData.CreateSequentialSampler());
            testing_writer.WriteSeparated(", ", "Beginning", report.TestsPassedCount, report.TestsFailedCount, report.Loss.Average, report.Loss.Max, report.Loss.Min, report.Accuracy, report.Precision, report.Recall, report.F1, testTimer.Elapsed.TotalSeconds);
            testTimer.Stop();
        }

        var totalTimer = Stopwatch.StartNew();
        var epochTimer = Stopwatch.StartNew();

        // Print headers for table
        WriteHeaders("Epoch", "Loss", "Accuracy", "Precision", "Recall", "Time (s)");

        // Do training loop
        // TODO progress bar (REQUIRED!)
        var lineStart = Console.GetCursorPosition();
        IProgress<ModuleTrainingEnumerator.EpochProgress>? progress = new ProgressAction<ModuleTrainingEnumerator.EpochProgress>(
            (progress) =>
            {
                Console.SetCursorPosition(lineStart.Left, lineStart.Top);
                var str = progress.Epoch.ToString();
                var width = Math.Max(0, Console.BufferWidth - str.Length);
                Console.Write(str);
                Console.Write(Qkmaxware.Terminal.Elements.ProgressBar.ToString(progress.Completed, width));
            }
        );
        while (session.MoveNext(progress))
        {
            var report = session.Current;
            var time = epochTimer.Elapsed.TotalSeconds;

            // Write to terminal
            Console.SetCursorPosition(lineStart.Left, lineStart.Top);
            WriteRow(report.Epoch, report.Loss.Average, report.Accuracy, report.Precision, report.Recall, time);
            lineStart = Console.GetCursorPosition();

            // Write validation report
            validation_writer.WriteSeparated(", ", report.Epoch, report.TestsPassedCount, report.TestsFailedCount, report.Loss.Average, report.Loss.Max, report.Loss.Min, report.Accuracy, report.Precision, report.Recall, report.F1, time);

            // TODO performance metrics? not currently being used anymore

            // Do retention
            if (retention_policies.Count > 0)
            {
                var st = new SafetensorSerializer();
                if (network is IBlockVisitable visitable)
                    visitable.Accept(st);

                foreach (var retention in retention_policies)
                {
                    retention.Backup($"epoch-{report.Epoch}.safetensors", st.ToSafetensors());
                }
            }

            // Send notification
            if (IsSet(NotificationEvent.step))
                notifier?.NotifyTrainingStep(network, report.Epoch, trainer.MaxEpochs, report);

            epochTimer.Restart();
        }
        totalTimer.Stop();
        epochTimer.Stop();

        if (TestingData is not null)
        {
            var testTimer = Stopwatch.StartNew();
            var report = session.Test(TestingData.CreateSequentialSampler());
            testing_writer.WriteSeparated(", ", "End", report.TestsPassedCount, report.TestsFailedCount, report.Loss.Average, report.Loss.Max, report.Loss.Min, report.Accuracy, report.Precision, report.Recall, report.F1, testTimer.Elapsed.TotalSeconds);
            testTimer.Stop();
        }

        if (IsSet(NotificationEvent.done))
            notifier?.NotifyTrainingDone(network, session.Current.Epoch, session.Current);
        #endregion
    }

    private int AutoPickBatchSize()
    {
        if (BatchSize.HasValue)
        {
            return Math.Max(1, BatchSize.Value);
        }
        return Math.Max(1, Environment.ProcessorCount);
    }

    private LossFunction AutoPickLoss(ModelInfo info, INetworkModule module)
    {
        // User provided loss function, use that if it matches one
        if (!string.IsNullOrEmpty(LossFunctionName))
        {
            var loss_function = LossFunctions
                .EnumerateAll()
                .Where(@delegate => @delegate.Name.Contains(LossFunctionName, StringComparison.CurrentCultureIgnoreCase))
                .FirstOrDefault();

            if (loss_function is not null)
                return loss_function;
        }

        // No user provided loss, auto-pick
        return SmartPickLoss(info);
    }

    public static LossFunction SmartPickLoss(ModelInfo info)
    {
        if (info.ProblemDescription?.Classification is not null)
            return LossFunctions.CategoricalCrossEntropy;

        return LossFunctions.MeanSquaredError;
    }

    private IInitializer AutoPickInitializer(ModelInfo info, INetworkModule module)
    {
        // User provided loss function, use that if it matches one
        if (!string.IsNullOrEmpty(InitializerName))
        {
            var initalizer = Initializers
                .EnumerateAll()
                .Where(init => init.GetType().Name.Contains(InitializerName, StringComparison.CurrentCultureIgnoreCase))
                .FirstOrDefault();

            if (initalizer is not null)
                return initalizer;
        }

        // Auto pick
        var walker = new ActivationCountWalker();
        if (module is IBlockVisitable visitable)
            visitable.Accept(walker);

        var mostUsed = walker.MostUsedActivation();
        switch (mostUsed)
        {
            // ReLU-family activations
            case Type lu when
                   lu == typeof(ReLU)
                || lu == typeof(LeakyReLU)
                || lu == typeof(ExponentialLU)
                || lu == typeof(GELU)
                || lu == typeof(PReLU):
                return Initializers.He;

            // TODO
            // Self-normalizing family activations 
            // case Type cu when cu == typeof(SELU):
            //      return Initializers.LeCun;

            // Step activations
            case Type xa when
                   xa == typeof(HyperbolicTangent)
                || xa == typeof(Sigmoid)
                || xa == typeof(Softplus):
                return Initializers.NormalXavier;

            // Fallback
            default:
                return Initializers.NormalXavier;
        }
    }

    public class ActivationCountWalker : BlockWalker
    {
        private Dictionary<Type, int> activationCounts = new Dictionary<Type, int>();
        public override void VisitModule(INetworkModule module)
        {
            if (module is not Activation act)
                return;

            var type = act.ActivationFunction.GetType();
            int count = 0;
            activationCounts.TryGetValue(type, out count);

            activationCounts[type] = count + 1;
        }
        public Type? MostUsedActivation() => activationCounts.OrderByDescending(x => x.Value).Select(x => x.Key).FirstOrDefault();
    }

    private INotifier? GetNotifierFor(string? endpoint)
    {
        var notifiers = typeof(Fit).Assembly
            .GetExportedTypes()
            .Where(type => !type.IsAbstract && type.IsClass && type.IsAssignableTo(typeof(INotifierFactory))
                && type.GetConstructor(Type.EmptyTypes) is not null
            )
            .Select(type => (ConstructorInfo?)type.GetConstructor(Type.EmptyTypes))
            .Cast<ConstructorInfo>()
            .Select(cons => (INotifierFactory)cons.Invoke(null))
            .ToList();

        if (string.IsNullOrEmpty(endpoint))
            return null;
        foreach (var notifier in notifiers)
        {
            if (notifier.SupportsEndpoint(endpoint))
                return notifier.Make(endpoint);
        }
        return null;
    }

    public static ITrainingDataSource<float> ReadTrainingData(string from)
    {
        var formats = typeof(Fit).Assembly
            .GetExportedTypes()
            .Where(type => !type.IsAbstract && type.IsClass && type.IsAssignableTo(typeof(ITrainingDataFormat))
                && type.GetConstructor(Type.EmptyTypes) is not null
            )
            .Select(type => (ConstructorInfo?)type.GetConstructor(Type.EmptyTypes))
            .Cast<ConstructorInfo>()
            .Select(cons => (ITrainingDataFormat)cons.Invoke(null))
            .ToList();

        foreach (var format in formats)
        {
            if (format.IsInFormat(from))
            {
                return format.Read(from);
            }
        }
        throw new FormatException($"Unknown file format for '{from}'");
    }

    private static ITrainingDataSource<float> Subsample(ITrainingDataSource<float> primary, float percent, int minAmount)
    {
        var count = primary.Count;
        if (count == 0)
            return primary;

        var subsamples = Math.Min(count, Math.Max(percent * count, minAmount));
        ListTrainingDataSource<float> sub = new ListTrainingDataSource<float>(primary.InputShape, primary.OutputShape);
        if (subsamples <= 0)
            return sub;

        var rng = Random.Shared;
        for (var i = 0; i < subsamples; i++)
        {
            var index = rng.Next(count);
            sub.Add(primary[index]);
        }
        return sub;
    }

    private bool IsSet(NotificationEvent evt) {
        return (this.NotificationTrigger & ((int)evt)) != 0;
    }

    private IElement LabeledText(string label, string text)
    {
        return new VSplitContainer(label.Length + 2, new Label(label + ":"), new Paragraph(text));
    }

    private void WriteHeaders(params ReadOnlySpan<string> headers)
    {
        var width = Console.BufferWidth;
        var widthPerColumn = ((width - headers.Length) / headers.Length);

        if (width < 3)
            return;

        // Write top header
        Console.Write('┌');
        for (var c = 0; c < headers.Length; c++)
        {
            if (c != 0)
                Console.Write('┬');

            for (var i = 0; i < widthPerColumn; i++)
            {
                Console.Write('─');
            }
        }
        Console.Write('┐');
        Console.WriteLine();

        // Write actual headers
        Console.Write('│');
        for (var c = 0; c < headers.Length; c++)
        {
            Console.Write(EnsureLength(headers[c], widthPerColumn));
            Console.Write('│');
        }
        Console.WriteLine();

        // Write footer
        Console.Write('├');
        for (var c = 0; c < headers.Length; c++)
        {
            if (c != 0)
                Console.Write('┼');

            for (var i = 0; i < widthPerColumn; i++)
            {
                Console.Write('─');
            }
        }
        Console.Write('┤');
        Console.WriteLine();
    }
    private void WriteRow(params ReadOnlySpan<object?> data)
    {
        var width = Console.BufferWidth;
        var widthPerColumn = ((width - data.Length) / data.Length);

        if (width < 3)
            return;

        Console.Write('│');
        for (var c = 0; c < data.Length; c++)
        {
            Console.Write(EnsureLength(data[c]?.ToString() ?? string.Empty, widthPerColumn));
            Console.Write('│');
        }
        Console.WriteLine();
    }
    private string EnsureLength(string str, int len)
    {
        if (str.Length < len)
            return str.PadRight(len, ' ');
        else if (str.Length > len)
            return str.Substring(0, len);
        else
            return str;
    }
}
