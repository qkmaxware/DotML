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

namespace DotML.Cli.Commands;

//Example usage
/*
dotml fit net2 --data-training training-pairs.bin --data-validation valid-pairs.bin --init he --weights none --profile --report
*/

// Example output
/*
|-------------------------------------|------------------|
| Training Data                       | Network Info     |
|     input shape: 3x224x224          |    arch: alexnet |
|     output shape: 1x10x1            |    layers: 25    |
|-------------------------------------|------------------|

Training 
| Epoch | Progress              | Accuracy | Loss |
|-------|-----------------------|----------|------| 
| 1     | [-------/--/---] 100% | 0.15     | 0.26 |
| 2     | [-------       ] 50%  |          |      |
*/

[Verb("fit", HelpText = "Train a compiled network against a provided training data set")]
public class Fit : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    [Option("data-training", HelpText = "The data-set used for training", Required = true)]
    public string? TrainingDataPath {get; set;}

    [Option("data-validation", HelpText = "The data-set used for validation", Required = false)]
    public string? ValidationDataPath {get; set;}

    [Option("data-test", HelpText = "The data-set used for testing", Required = false)]
    public string? TestingDataPath {get; set;}

    [Option("max-epochs", HelpText = "Maximum number of epochs to train for (min 1)", Required = false, Default = 100)]
    public int MaxEpochs {get; set;}

    [Option("learning-rate", HelpText = "Initial learning rate", Required = false, Default = 0.01)]
    public double LearningRate {get; set;}

    [Option("accuracy", HelpText = "Early stop accuracy threshold", Required = false, Default = 0.15)]
    public double Accuracy {get; set;}

    [Option("patience", HelpText = "Early stop patience (min 1)", Required = false, Default = 1)]
    public int Patience {get; set;}

    [Option("batch-size", HelpText = "Batch size, leave blank for batch to be automatically determined", Required = false)]
    public int? BatchSize {get; set;}

    [Option("loss", HelpText = "Name of the loss function to use, leave blank for function to be automatically determined", Required = false)]
    public string? LossFunctionName {get; set;}

    [Option("clip-gradients", HelpText = "Flag to indicate if gradient clipping should be performed", Required = false, Default = "false")]
    public string? ClipGradientsStr {get; set;}
    public bool ClipGradients => IsSet(ClipGradientsStr);

    [Option("skip-testing", HelpText = "Flag to indicate if the testing phase should be skipped", Required = false, Default = "false")]
    public string? SkipTestingStr {get; set;}
    public bool SkipTesting => IsSet(SkipTestingStr);

    [Option("continue", HelpText = "Flag to indicate if the existing weights should be used or if the network should be re-initialized", Required = false, Default = "false")]
    public string? ContinueStr {get; set;}
    public bool ContinueFromExisting => IsSet(ContinueStr);

    [Option("retry", HelpText = "Number of times to retry training if an exception occurs", Required = false, Default = 0)]
    public int? RetryCount {get; set;}

    public enum UpdateModeType {
        none, overwrite, duplicate
    }
    [Option("save", HelpText = "Flag to indicate how the final weights should be applied to the model (none, overwrite, or duplicate)", Default = UpdateModeType.overwrite)]
    public UpdateModeType UpdateMode {get; set;}

    public enum RetentionPolicy {
        none,
        all, 
        most_recent,
        last5,
        last10,
        smallest_loss,
        highest_accuracy,
        most_passed,
    }
    [Option("retention", HelpText = "Flag to indicate how intermediate weights should be retained (none, all, most_recent, last5, last10, smallest_loss)", Default = RetentionPolicy.none)]
    public RetentionPolicy Retention {get; set;}
    

    [Option("notify", HelpText = "Url, endpoint, or address to send notification updates to.", Required = false, Default = null)]
    public string? NotifyEndpoint {get; set;}

    [Option("notify-on", HelpText = "A list of events which trigger notifications", Required = false, Default = new NotificationEvent[]{ NotificationEvent.all })]
    public IEnumerable<NotificationEvent>? NotificationEvents {get; set;}
    private int NotificationTrigger {
        get {
            int trigger = 0;
            if (NotificationEvents is not null) {
                foreach (var evt in NotificationEvents) {
                    trigger |= (int)evt;
                }
            }
            return trigger;
        }
    }

    public override void Action(AppData appData) {
        #region Validate Args
        var model = appData.GetModel(ModelName);
        if (model is null) {
            Console.WriteLine($"No model exists with name '{ModelName}'.");
            return;
        }
        FileInfo training_file;
        if (string.IsNullOrEmpty(TrainingDataPath) || !((training_file = new FileInfo(TrainingDataPath)).Exists)) {
            Console.WriteLine($"The training data file '{TrainingDataPath}' doesn't exist.");
            return;
        }
        FileInfo? validation_file = null;
        if (!string.IsNullOrEmpty(ValidationDataPath) && !((validation_file = new FileInfo(ValidationDataPath)).Exists)) {
            Console.WriteLine($"The validation data file '{TrainingDataPath}' doesn't exist.");
            return;
        }
        FileInfo? testing_file = null;
        if (!string.IsNullOrEmpty(TestingDataPath) && !((testing_file = new FileInfo(TestingDataPath)).Exists)) {
            Console.WriteLine($"The testing data file '{TestingDataPath}' doesn't exist.");
            return;
        }
        INotifier? notifier = GetNotifierFor(NotifyEndpoint);

        var dir = appData.CreateTrainingDir();
        var weights_dir = Path.Combine(dir.FullName, "weights");
        Directory.CreateDirectory(weights_dir);
        using (var writer = new StreamWriter(Path.Combine(dir.FullName, $"model.{model.Guid}.xml"))) {
            writer.Write(model.ToXml());
        }
        #endregion

        #region Network
        var network = model.Load();
        if (network is IMarkdownable md) {
            using (var writer = new StreamWriter(Path.Combine(dir.FullName, "network-description.md"))) {
                md.ToMarkdown(writer);
            }
        } else if (network is IJsonizable json) {
            using (var writer = new StreamWriter(Path.Combine(dir.FullName, "network-description.json"))) {
                json.ToJson(writer);
            }
        } else if (network is IHtmlable html) {
            using (var writer = new StreamWriter(Path.Combine(dir.FullName, "network-description.html"))) {
                html.ToHtml(writer);
            }
        } 
        
        if (network is IDiagrammable svg) {
            using (var writer = new StreamWriter(Path.Combine(dir.FullName, "network-diagram.svg"))) {
                svg.ToSvg(writer);
            }
        }
        #endregion

        #region Trainer
        // TODO make all of these things configurable
        var testing_report      = new DefaultValidationReport();
        var validation_report   = new DefaultValidationReport();
        var performance_report  = new DefaultProfilingReport();
        var trainer             = new EnumerableBatchTrainer<FeedforwardNetwork> {
            Epochs                  = Math.Max(1, MaxEpochs),
            LearningRate            = Math.Max(0, LearningRate),
            LearningRateOptimizer   = new AdamOptimizer(),
            EarlyStop               = true,
            EarlyStopAccuracy       = Math.Max(0, Accuracy),
            EarlyStopPatience       = Math.Max(1, Patience),
            LossFunction            = smart_pick_loss(network),
            NetworkInitializer      = smart_pick_initializer(network),
            BatchSize               = smart_pick_batch_size(network),
            EnableGradientClipping  = ClipGradients,
            ClippingThresholdSynapses = 10,
            ClippingThresholdBiases = 5.0,
            ValidationReport        = validation_report,
            Profiler                = performance_report
        };
        using (var trainer_prop_writer = new StreamWriter(Path.Combine(dir.FullName, "trainer-config.yaml"))) {
            trainer_prop_writer.WriteLine("Trainer:");
            trainer_prop_writer.Write("    "); trainer_prop_writer.Write("Type"); trainer_prop_writer.Write(": "); trainer_prop_writer.WriteLine(trainer.GetType().Name);
            foreach (PropertyInfo property in trainer.GetType().GetProperties()) {
                object? value = property.CanRead ? property.GetValue(trainer, null) : null;
                if (value is LossFunction loss)
                    value = loss.Name;
                else 
                    value = value?.ToString() ?? "n/a";
                trainer_prop_writer.Write("    "); trainer_prop_writer.Write(property.Name); trainer_prop_writer.Write(": "); trainer_prop_writer.WriteLine(value);
            }
        }
        IRetentionPolicy<Safetensors> retention_policy = this.Retention switch {
            RetentionPolicy.all             => new AllWeights(weights_dir),
            RetentionPolicy.most_recent     => new MostRecentWeights(weights_dir),
            RetentionPolicy.last5           => new LastNWeights(weights_dir, 5),
            RetentionPolicy.last10          => new LastNWeights(weights_dir, 10),
            RetentionPolicy.smallest_loss   => new SmallestLoss(weights_dir, validation_report),
            RetentionPolicy.highest_accuracy=> new HighestAccuracy(weights_dir, validation_report),
            RetentionPolicy.most_passed     => new MostTestsPassed(weights_dir, validation_report),
            _                               => new NoWeights()
        };
        #endregion

        #region Data
        var randgen = new Random();
        TrainingSet trainingPairs   = ReadData(training_file);                                                      // Data used in backpropagation
        TrainingSet validationPairs = validation_file is not null ? ReadData(validation_file) : new TrainingSet(trainingPairs.SampleRandomly((int)Math.Max(1, 0.25 * trainingPairs.Size)).AsEnumerable());    // Data used in early-stop & validation
        TrainingSet testingPairs    = testing_file is not null ? ReadData(testing_file) : trainingPairs;             // Data used in verify model "generality"
        if (trainingPairs.Size != 0) {
            var first = trainingPairs[0];
            if (first.Input.Dimensionality != network.InputShape.Count) {
                Console.WriteLine($"The input shape of the network ({network.InputShape.Count}={network.InputShape}) does not match the input shape of the training data ({first.Input.Dimensionality}).");
                return;
            }
            if (first.Output.Dimensionality != network.OutputShape.Count) {
                Console.WriteLine($"The output shape of the network ({network.OutputShape.Count}={network.OutputShape}) does not match the output shape of the training data ({first.Output.Dimensionality}).");
                return;
            }
        }
        if (validationPairs.Size != 0) {
            var first = validationPairs[0];
            if (first.Input.Dimensionality != network.InputShape.Count) {
                Console.WriteLine($"The input shape of the network ({network.InputShape.Count}={network.InputShape}) does not match the input shape of the validation data ({first.Input.Dimensionality}).");
                return;
            }
            if (first.Output.Dimensionality != network.OutputShape.Count) {
                Console.WriteLine($"The output shape of the network ({network.OutputShape.Count}={network.OutputShape}) does not match the output shape of the validation data ({first.Output.Dimensionality}).");
                return;
            }
        }
        if (testingPairs.Size != 0) {
            var first = testingPairs[0];
            if (first.Input.Dimensionality != network.InputShape.Count) {
                Console.WriteLine($"The input shape of the network ({network.InputShape.Count}={network.InputShape}) does not match the input shape of the testing data ({first.Input.Dimensionality}).");
                return;
            }
            if (first.Output.Dimensionality != network.OutputShape.Count) {
                Console.WriteLine($"The output shape of the network ({network.OutputShape.Count}={network.OutputShape}) does not match the output shape of the testing data ({first.Output.Dimensionality}).");
                return;
            }
        }
        var batch_size              = trainer.BatchSize;
        var batch_count             = (trainingPairs.Size + trainer.BatchSize - 1) / trainer.BatchSize;
        var validation_batch_count  = (validationPairs.Size + trainer.BatchSize - 1) / trainer.BatchSize;
        var iteration_count         = batch_count + validation_batch_count;
        #endregion

        // TODO print header
        string[] main_headers = ["NETWORK", "TRAINING-DATA"];
        int[] main_headers_len = [60, 40];
        for (var col = 0; col < main_headers.Length; col++) {
            var name = main_headers[col];
            var len = main_headers_len[col];
            Console.Write(ColumnValue(name, len));
            Console.Write(' ');
        }
        Console.WriteLine();

        Console.Write(ColumnValue($"Name: {network.Name}", main_headers_len[0]));
        Console.Write(' ');
        Console.Write(ColumnValue($"Training Entries: {trainingPairs.Size}", main_headers_len[1]));
        Console.WriteLine();
        Console.Write(ColumnValue($"Layers: {network.LayerCount}", main_headers_len[0]));
        Console.Write(' ');
        Console.Write(ColumnValue($"Validation Entries: {validationPairs.Size}", main_headers_len[1]));
        Console.WriteLine();
        Console.Write(ColumnValue($"Input Shape: {network.InputShape}", main_headers_len[0]));
        Console.Write(' ');
        Console.Write(ColumnValue($"Testing Entries: {testingPairs.Size}", main_headers_len[1]));
        Console.WriteLine();
        Console.Write(ColumnValue($"Output Shape: {network.OutputShape}", main_headers_len[0]));
        Console.Write(' ');
        Console.Write(ColumnValue($"Input Size: {(trainingPairs.Size > 0 ? trainingPairs[0].Input.Dimensionality : 0)}", main_headers_len[1]));
        Console.WriteLine();
        Console.Write(ColumnValue($"Size: {network.StorageSize()}", main_headers_len[0]));
        Console.Write(' ');
        Console.Write(ColumnValue($"Output Size: {(trainingPairs.Size > 0 ? trainingPairs[0].Output.Dimensionality : 0)}", main_headers_len[1]));
        Console.WriteLine();
        Console.WriteLine();

        Console.WriteLine("TRAINER");
        foreach (var property in trainer.GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public)) {
            if (!property.CanRead)
                continue;
            Console.WriteLine($"{property.Name}: {property.GetValue(trainer)}");
        }
        Console.WriteLine();

        Console.WriteLine();
        string[] training_headers = ["EPOCH", "PROGRESS", "ACCURACY", "PRECISION", "RECALL", "LOSS", "VALIDATION", "TIME-TAKEN"];
        int[] training_header_len = [5,       25,         10,          10,          10,       10,    15,            25          ];
        var divider_len = training_header_len.Sum();
        Console.WriteLine(new String('-', divider_len));
        Console.WriteLine();

        int retries = RetryCount.HasValue ? Math.Max(0, RetryCount.Value) : 0; // No retries by default
        int attempt = 1;
        System.Diagnostics.Stopwatch? start_timer = null;
        training_start_label:
        try {

        #region Training
        var session = trainer.EnumerateTraining(network, trainingPairs.SampleRandomly(), validationPairs.SampleSequentially());
        session.Reset();

        #region Training / Load checkpoint
        // TODO load checkpoint / prior weights
        if (ContinueFromExisting) {
            Console.WriteLine($"Using pre-trained weights: '{model.GetWeightsFile()}'");
            network.FromSafetensor(model.FetchSavedWeights()); // Undo the "reset" operation on the weights
        }
        #endregion

        for (var col = 0; col < training_headers.Length; col++) {
            var name = training_headers[col];
            var len = training_header_len[col];
            Console.Write(ColumnValue(name, len));
            Console.Write(' ');
        }
        Console.WriteLine();

        ProgressBar? current_progress = null;
        //var has_cancelled = false;
        session.OnEpochStart += (int epoch, int epochCount) => {
            current_progress?.Update(0, iteration_count);
            
        };
            session.OnBatchStart += (int batch, int batchCount) => {
                
            };
            session.OnBatchEnd += (int batch, int batchCount) => {
                current_progress?.Update(batch, iteration_count);
                
            };
            session.OnValidationStart += (int epoch, int epochCount) => {
                current_progress?.Update(batch_count, iteration_count);
                
            };
                session.OnValidated += (int epoch, int epochCount, int inputIndex, double loss) => {
                    var actual_progress = batch_count + inputIndex/batch_size; // batch index for validation
                    current_progress?.Update(actual_progress, iteration_count);
                    
                };
            session.OnValidationEnd += (int epoch, int epochCount, double loss) => {
                
            };
        session.OnEpochEnd += (int epoch, int epochCount) => {
           
        };

        using var validation_writer = new StreamWriter(Path.Combine(dir.FullName, "validation.csv"));
        validation_writer.WriteLine("Epoch, Tests-Passed, Tests-Failed, Loss-Average, Loss-Max, Loss-Min, Accuracy, Precision, Recall, F1-Score, Time-Taken");
        validation_writer.Flush();

        using var testing_writer = new StreamWriter(Path.Combine(dir.FullName, "testing.csv"));
        testing_writer.WriteLine("Epoch, Tests-Passed, Tests-Failed, Loss-Average, Loss-Max, Loss-Min, Accuracy, Precision, Recall, F1-Score, Time-Taken");
        testing_writer.Flush();
        
        start_timer = Stopwatch.StartNew();
        (double accuracy, double precision, double recall, double minloss, double maxloss, double avgloss, int passed)? prev_report = null;
        var best_loss = double.MaxValue;
        var has_next = true;
        Console.CancelKeyPress += delegate (object? sender, ConsoleCancelEventArgs e) {
            has_next = false;                   // Stop at end of next iteration
            //has_cancelled = true;               // Indicate that we have cancelled it
            UpdateMode = UpdateModeType.none;   // We don't want to preserve weights if cancelled
            Console.WriteLine(); Console.WriteLine();
            Console.WriteLine("<!-- Training Cancelled by User -->");
            DrawDivider(divider_len);
            PrintDone(dir);
            if (IsSet(NotificationEvent.cancelled)) {
                notifier?.NotifyTrainingCancelled(network, session.CurrentEpoch);
            }
        };
        if (IsSet(NotificationEvent.started)) {
            notifier?.NotifyTrainingStarted(network);
        }
        
        // Do an initial status test to see "how good it is" originally
        validation_report.Reset();
        Test(network, testingPairs, validation_report, trainer.BatchSize, trainer.EarlyStopAccuracy, trainer.LossFunction);
        validation_writer.WriteLine($"{0}, {validation_report.TestsPassedCount}, {validation_report.TestsFailedCount}, {validation_report.AverageLoss}, {validation_report.MaxLoss}, {validation_report.MinLoss}, {validation_report.Accuracy}, {validation_report.Precision}, {validation_report.Recall}, {validation_report.F1Score}, \"n/a\"");
        validation_writer.Flush();
        validation_report.Reset();
        if (!SkipTesting && testing_report is not null) {
            var report = testing_report;
            Test(network, testingPairs, report, trainer.BatchSize, trainer.EarlyStopAccuracy, trainer.LossFunction);

            testing_writer.WriteLine($"{0}, {report.TestsPassedCount}, {report.TestsFailedCount}, {report.AverageLoss}, {report.MaxLoss}, {report.MinLoss}, {report.Accuracy}, {report.Precision}, {report.Recall}, {report.F1Score}, \"n/a\"");
            testing_writer.Flush();
        }

        while (has_next) {
            #region Training / Epoch Start
            // Print the beginning of the epoch entry to the CLI
            var epoch_id = session.CurrentEpoch + 1;
            Console.Write(ColumnValue(epoch_id, training_header_len[0]));
            Console.Write(' '); 
            #endregion

            #region Training / Epoch Step
            // Print the training progress for the epoch to the CLI
            current_progress = new ProgressBar(training_header_len[1] - 5);
            current_progress.Mark(batch_count, iteration_count); // Create a mark at the given divider between batching and validation
            var timer = Stopwatch.StartNew();
            has_next = session.MoveNext();
            timer.Stop();
            var elapsed = timer.Elapsed;
            current_progress.Update(1.0);
            Console.Write(' ');
            #endregion

            #region Training / Epoch End
            // Print the rest of the epoch entry to the CLI 
            var report = validation_report;
            if (IsSet(NotificationEvent.step)) {
                notifier?.NotifyTrainingStep(network, epoch_id, trainer.Epochs, validation_report);
            }
            var def_colour = Console.ForegroundColor;
            Console.ForegroundColor = !prev_report.HasValue ? def_colour : (prev_report.Value.accuracy <= report.Accuracy ? ConsoleColor.Green : ConsoleColor.Red); // Accuracy should be higher
            Console.Write(ColumnValue(report.Accuracy, training_header_len[2]));
            Console.Write(' '); 
            Console.ForegroundColor = !prev_report.HasValue ? def_colour : (prev_report.Value.precision <= report.Precision ? ConsoleColor.Green : ConsoleColor.Red); // Precision should be higher
            Console.Write(ColumnValue(report.Precision, training_header_len[3]));
            Console.Write(' '); 
            Console.ForegroundColor = def_colour;
            Console.Write(ColumnValue(report.Recall, training_header_len[4]));
            Console.Write(' '); 
            var isBetterAverageLoss = prev_report.HasValue && report.AverageLoss < prev_report.Value.avgloss;
            Console.ForegroundColor = !prev_report.HasValue ? def_colour : (prev_report.Value.avgloss >= report.AverageLoss ? ConsoleColor.Green : ConsoleColor.Red); // Loss should be smaller
            Console.Write(ColumnValue(report.AverageLoss, training_header_len[5]));
            Console.Write(' '); 
            Console.ForegroundColor = !prev_report.HasValue ? def_colour : (prev_report.Value.passed <= report.TestsPassedCount ? ConsoleColor.Green : ConsoleColor.Red); // Passed should be higher
            Console.Write(ColumnValue(report.TestsPassedCount + "/" + report.TestCount, training_header_len[6]));
            Console.Write(' '); 
            Console.ForegroundColor = def_colour;
            Console.Write(ColumnValue(elapsed.TotalMinutes + "m", training_header_len[7]));
            Console.WriteLine();
            prev_report = (report.Accuracy, report.Precision, report.Recall, report.MinLoss, report.MaxLoss, report.AverageLoss, report.TestsPassedCount);

            // Save validation report entry
            validation_writer.WriteLine($"{epoch_id}, {report.TestsPassedCount}, {report.TestsFailedCount}, {report.AverageLoss}, {report.MaxLoss}, {report.MinLoss}, {report.Accuracy}, {report.Precision}, {report.Recall}, {report.F1Score}, \"{elapsed}\"");
            validation_writer.Flush();

            if (IsSet(NotificationEvent.best) && (report.AverageLoss < best_loss)) {
                notifier?.NotifyNewBest(network, epoch_id, trainer.Epochs, validation_report);
            }
            best_loss = Math.Min(report.AverageLoss, best_loss);

            // Save testing report entry (currently no UI to monitor this)
            if (!SkipTesting && testing_report is not null) {
                report = testing_report;
                Test(network, testingPairs, report, trainer.BatchSize, trainer.EarlyStopAccuracy, trainer.LossFunction);

                testing_writer.WriteLine($"{epoch_id}, {report.TestsPassedCount}, {report.TestsFailedCount}, {report.AverageLoss}, {report.MaxLoss}, {report.MinLoss}, {report.Accuracy}, {report.Precision}, {report.Recall}, {report.F1Score}, \"{elapsed}\"");
                testing_writer.Flush();
            }

            // Save performance metrics, always re-write and not append (unlike the validation report)
            if (performance_report is not null) {
                using (var performance_writer = new StreamWriter(Path.Combine(dir.FullName, "performance.csv"))) {
                    performance_writer.WriteLine("Benchmark-Name, Time-Min (s), Time-Max (s), Time-Average (s), Time-Total (s), Sample-Count");
                    
                    foreach (var metric in performance_report.Benchmarks.OrderBy(x => x.Name)) {
                        performance_writer.WriteLine($"{metric.Name}, {metric.Min.TotalSeconds}, {metric.Max.TotalSeconds}, {metric.Average.TotalSeconds}, {metric.Sum.TotalSeconds}, {metric.Count}");
                    }
                }
            }

            // Save weights
            try {
                retention_policy.Backup($"epoch-{epoch_id}.safetensors", network.ToSafetensor());
                //network.ToSafetensor().WriteToFile(Path.Combine(weights_dir,  $"epoch-{epoch_id}.safetensors"));
            } catch {}
            #endregion
        }
        if (IsSet(NotificationEvent.done)) {
            notifier?.NotifyTrainingDone(network, session.CurrentEpoch, validation_report);
        }
        start_timer.Stop();
        } catch (Exception e) {
            using (var writer = new StreamWriter(File.Open(Path.Combine(dir.FullName, "errors.log"), FileMode.Append))) {
                writer.WriteLine($"Attempt {attempt}");
                writer.WriteLine(e);
                writer.WriteLine();
            }

            if (retries <= 0) {
                throw;
            } else {
                Console.WriteLine(); Console.WriteLine();
                Console.WriteLine("<!-- Retrying After Exception -->");
                Console.WriteLine();
                retries--;
                attempt++;
                goto training_start_label; // Just try the training all over again from the beginning, but with one less retry
            }
        }
        #region Training / Done

        #endregion
        #endregion

        #region Cleanup
        DrawDivider(divider_len);
        switch (UpdateMode) {
            case UpdateModeType.none:
                break;
            case UpdateModeType.duplicate:
                model = new ModelInfo(model);
                goto case UpdateModeType.overwrite;
            case UpdateModeType.overwrite:
                model.UpdateWeights(network.ToSafetensor());

                var training_report = (!SkipTesting && testing_report is not null) ? testing_report : validation_report;
                model.TrainingMetadata = new ModelTrainingInfo {
                    TrainingDuration = start_timer is null ? null : start_timer.Elapsed.ToString("G"),
                    Accuracy = training_report.Accuracy,
                    Precision = training_report.Precision,
                    Recall = training_report.Recall,
                    MaxLoss = training_report.MaxLoss,
                    MinLoss = training_report.MinLoss,
                    AvgLoss = training_report.AverageLoss,
                };
                model.UpdateMetadata();
                break;
        }
        PrintDone(dir);
        #endregion
    }

    private void PrintDone(DirectoryInfo training_dir) {
        Console.WriteLine($"Reports saved to '{training_dir.FullName}'.");
        Console.WriteLine($"Use '{typeof(Fit).Assembly.GetName().Name} reports open '{training_dir.Name}' to review training metrics.");
    }

    #region Utility Functions
    private LossFunction smart_pick_loss(FeedforwardNetwork network) {
        // User provided loss function, use that if it matches one
        if (!string.IsNullOrEmpty(LossFunctionName)) {
            var loss_function = LossFunctions
                .EnumerateAll()
                .Where(@delegate => @delegate.Name.Contains(LossFunctionName, StringComparison.CurrentCultureIgnoreCase))
                .FirstOrDefault();
                
            if (loss_function is not null)
                return loss_function;
        }

        // Softmax layer use cross-entropy
        if (network.GetOutputLayer() is SoftmaxLayer) {
            // Remove softmax for training as Categorical Cross Entropy already does softmax
            //var removed = network.RemoveLayer(network.LayerCount - 1);
            //if (removed is not SoftmaxLayer)
                //throw new Exception("Failed to remove softmax layer from network for use with categorical cross entropy");
            return LossFunctions.CategoricalCrossEntropy;
        }

        // Non-softmax layer
        return LossFunctions.MeanSquaredError;
    }
    private int smart_pick_batch_size(FeedforwardNetwork network) {
        if (BatchSize.HasValue) {
            return Math.Max(1, BatchSize.Value);
        }
        return Math.Max(1, Environment.ProcessorCount);
    }
    private IInitializer smart_pick_initializer(FeedforwardNetwork network) {
        Dictionary<Type, int> counts = new Dictionary<Type, int>();
        for(var l = 0; l < network.LayerCount; l++) {
            var layer = network.GetLayer(l);
            if (layer is not ActivationLayer activation)
                continue;

            var func = activation.ActivationFunction.GetType();
            if (!counts.ContainsKey(func)) {
                counts[func] = 1;
            } else {
                counts[func]++;
            }
        }
        var most_used = counts.OrderByDescending(x => x.Value).Select(x => x.Key).FirstOrDefault();
        if (most_used is null) {
            return new NormalXavierInitialization();
        }

        if (most_used == typeof(HyperbolicTangent) || most_used == typeof(Sigmoid)) {
            return new NormalXavierInitialization();
        } else {
            return new HeInitialization();
        }
    }
    public static void Test(FeedforwardNetwork network, TrainingSet data, IValidationReport report, int batch_size, double pass_threshold, LossFunction loss_fn) {
        var data_iterator = data.SampleSequentially(); 
        var max_error = double.MinValue;
        var all_less_threshold = true;
        report.Reset();
        List<(FeatureSet<double> InMatrix, Vec<double> In, Vec<double> Out)> batch = new List<(FeatureSet<double> InMatrix, Vec<double> In, Vec<double> Out)>();
        var concurrency_level = batch_size; // or Environment.ProcessorCount
        while (data_iterator.MoveNext() && batch.Count < concurrency_level) {
            var pair = data_iterator.Current;
            var input = new FeatureSet<double>(pair.Input.Shape(network.InputShape).ToArray());
            batch.Add((input, pair.Input, pair.Output));
        }
        var batch_input = new BatchedFeatureSet<double>(batch.Select(x => x.InMatrix).ToArray());

        while (batch.Count > 0) {
            // Perform Feed-Forward
            var batch_predicted = network.PredictSync(batch_input);

            // Measure loss across batch
            for (var batchIndex = 0; batchIndex < batch_input.Batches; batchIndex++) {
                var input = batch[batchIndex].In;
                var @true = batch[batchIndex].Out;
                var predicted =  Vec<double>.Wrap(batch_predicted[batchIndex].SelectMany(mtx => mtx.FlattenRows()).ToArray());
                
                var loss = loss_fn.Invoke(predicted, @true);
                max_error = Math.Max(max_error, loss);
                var passed = loss < pass_threshold;
                all_less_threshold &= passed;
                report.Append(input, @true, predicted, passed, loss);
            }

            // Update UI


            // Compute next batch
            batch.Clear();
            while (data_iterator.MoveNext() && batch.Count < concurrency_level) {
                var pair = data_iterator.Current;
                var input = new FeatureSet<double>(pair.Input.Shape(network.InputShape).ToArray());
                batch.Add((input, pair.Input, pair.Output));
            }
            batch_input = new BatchedFeatureSet<double>(batch.Select(x => x.InMatrix).ToArray());
        }
    }

    private static INotifierFactory[] notifiers = [
        new DiscordFactory(),
        new SlackFactory(),
        new FileFactory(),
    ];
    private INotifier? GetNotifierFor(string? endpoint) {
        if (string.IsNullOrEmpty(endpoint))
            return null;
        foreach (var notifier in notifiers) {
            if (notifier.SupportsEndpoint(endpoint))
                return notifier.Make(endpoint);
        }
        return null;
    }
    private bool IsSet(NotificationEvent evt) {
        return (this.NotificationTrigger & ((int)evt)) != 0;
    }

    private static ITrainingDataFormat[] formats = [
        new TrainingData.JsonVectorPairs(),
        new TrainingData.ClassifiedCsv(),
        new TrainingData.BinaryTrainingSet(),
        new TrainingData.BinaryClassifiedVectors()
    ];
    public static TrainingSet ReadData(FileInfo file) {
        foreach (var format in formats) {
            if (format.IsInFormat(file)) {
                return format.Read(file);
            }
        }
        throw new FormatException($"Unknown file format for '{file.Name}'");
    }
    #endregion

    #region Utility Classes
    private class TrainingCancelledEvent : Exception { }
    #endregion
}