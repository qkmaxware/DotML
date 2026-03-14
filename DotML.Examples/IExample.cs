using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Text.Json;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Examples;

public enum ExampleKind
{
    /// <summary>
    /// Classify inputs into one of several categories
    /// </summary>
    Classification,
    /// <summary>
    /// Predict the output of a continuous function
    /// </summary>
    Regression, 
    /// <summary>
    /// Group or cluster data points
    /// </summary>
    Clustering,
    /// <summary>
    /// Detect outliers in a dataset
    /// </summary>
    AnomalyDetection,
    /// <summary>
    /// Generate new data based on an input
    /// </summary>
    Generative,
}

public enum TrainingMethod
{
    /// <summary>
    /// Trained via gradient descent and backpropagation
    /// </summary>
    Backpropagation,
    /// <summary>
    /// Trained using genetic training
    /// </summary>
    Evolutionary,
    /// <summary>
    /// Trained via reinforcement
    /// </summary>
    Reinforcement,
    /// <summary>
    /// Trained by adversarial competition
    /// </summary>
    Adversarial
}

public interface IExample
{
    public string Name { get; }
    public void Configure(string json);

    public ExampleKind Kind {get;}
    public TrainingMethod TrainingMethod {get;}

    public bool HasBeenTrained();
    public string? GetDescription();
    public void Run(IEnumerable<string> inputs, string? outputPath);
    public void ProcessRawData();
    public void Train(bool useExistingWeights, bool useLogging, int? saveInterval);
    public void TrainAllVariations(bool useExistingWeights, bool useLogging, int? saveInterval);
    public void Validate(bool useLogging);
    public void Clean();
}

public abstract class Example : IExample
{
    public virtual string Name => GetType().Name;

    protected virtual string ExamplePath => Path.Combine(Environment.CurrentDirectory, "Examples", Name);
    protected virtual string ProcessedDataPath => Path.Combine(ExamplePath, "tensors");
    protected virtual string RawDataPath => Path.Combine(ExamplePath, "raws");

    public abstract ExampleKind Kind {get;}
    public abstract TrainingMethod TrainingMethod {get;}

    public virtual string? GetDescription() => null;

    protected bool TryParseConfigString<TData>(string dataOrPath, [NotNullWhen(true)] out TData? data) where TData:class
    {
        data = null;
        var trimmed = MemoryExtensions.Trim(dataOrPath);
        try
        {
            if (trimmed.StartsWith("{") && trimmed.EndsWith("}"))
            {
                data = JsonSerializer.Deserialize<TData>(trimmed);
                return data is not null;
            } else
            {
                using var stream = File.OpenRead(dataOrPath);
                data = JsonSerializer.Deserialize<TData>(stream);
                return data is not null;
            }
        } catch
        {
            return false;
        }
    }

    public abstract void Clean();

    public abstract void Configure(string json);

    public virtual void ProcessRawData() {}

    public virtual bool HasBeenTrained() => false;
    public abstract void Run(IEnumerable<string> inputs, string? outputPath);

    public abstract void Train(bool useExistingWeights, bool useLogging, int? saveInterval);
    public virtual void TrainAllVariations(bool useExistingWeights, bool useLogging, int? saveInterval) => Train(useExistingWeights, useLogging, saveInterval);
    public abstract void Validate(bool useLogging);

    private DateTime startTime = DateTime.Now;

    protected StreamWriter CreateLogger(string name)
    {
        return new StreamWriter(startTime.ToString("yyyy-MM-dd--HH-mm-ss") + "." + name);
    }
    
    protected void WriteRow(params ReadOnlySpan<object?> values)
    {
        var columnWidth = Console.BufferWidth / values.Length;
        for (var i = 0; i < values.Length; i++)
        {
            Console.Write(ToString(values[i], columnWidth));
        }
        Console.WriteLine();
    }
    
    private string ToString(object? obj, int width)
    {
        var str = obj?.ToString() ?? "null";
        if (str.Length < width)
            return str.PadRight(width, ' ');
        else if (str.Length == width)
            return str;
        else
            return str.Substring(0, width);
    }
}

/// <summary>
/// Examples that use ModuleTrainer for backpropagation based learning (most common)
/// </summary>
public abstract class BackpropExample : Example
{
    public override void Configure(string json) { }

    protected const string DefaultWeightsFilename = "Network.safetensors";

    protected void RestoreWeights(INetworkModule network, bool throws = false)
    {
        try
        {
            var tensors = this.LoadWeights();
            var applier = new SafetensorDeserializer();
            if (network is IBlockVisitable visitable)
                applier.Deserialize(visitable, tensors);
        }
        catch (Exception)
        {
            // Re-throw if configured to
            if (throws)
                throw;
        }
    }

    protected void SaveWeights(INetworkModule network, bool throws = false)
    {
        try
        {
            var applier = new SafetensorSerializer();
            if (network is IBlockVisitable visitable)
                applier.Serialize(visitable);
            this.SaveWeights(applier.ToSafetensors());
        }
        catch (Exception)
        {
            // Re-throw if configured to
            if (throws)
                throw;
        }
    }

    public override ExampleKind Kind => ExampleKind.Classification;
    public override TrainingMethod TrainingMethod => TrainingMethod.Backpropagation;

    public override void Run(IEnumerable<string> InputStrings, string? OutputPath)
    {
        // Load network
        var network = this.GetArchitecture();

        // Load weights (required)
        RestoreWeights(network, throws: true);

        // Parse user input
        if (InputStrings is null)
            return;

        using TextWriter pipe = !string.IsNullOrEmpty(OutputPath) 
            ? new StreamWriter(Path.ChangeExtension(OutputPath, EnforceExtension(Path.GetExtension(OutputPath))))
            : System.Console.Out;
        foreach (var str in InputStrings)
        {
            pipe.Write("> "); pipe.WriteLine(str);
            var input = this.ParseUserInput(str);
            var output = network.Forward(input);
            pipe.WriteLine(this.FormatOutput(str, input, output));
            pipe.WriteLine(); // Extra line between inputs
        }
    }

    protected string EnforceExtension(string ext) => ".txt"; // Always enforce txt unless an example requires something unique

    public override void Train(bool useExistingWeights, bool useLogging, int? saveInterval)
    {
        var exampleName = this.GetType().Name;

        // Load tensors (ensure that we have enough data)
        if (!Directory.Exists(ProcessedDataPath) == false)
        {
            Directory.CreateDirectory(ProcessedDataPath);
        }
        if (!Directory.Exists(RawDataPath) == false)
        {
            Directory.CreateDirectory(RawDataPath);
        }
        this.LoadTrainingData(out var training, out var validation);
        if (training.Count == 0)
        {
            throw new InvalidOperationException("Training data is not provided");
        }
        if (validation.Count == 0)
        {
            validation = training;
        }

        // Load network
        var network = this.GetArchitecture();

        // Setup trainer (sensible defaults)
        var trainer = new ModuleTrainer();

        this.ConfigureTrainer(trainer); // Example specific configs

        // Do training loop
        ModuleTrainingEnumerator session = (ModuleTrainingEnumerator)trainer.EnumerateTraining(
            network: network,
            dataset: this.GetTrainingSampler(training),
            validation: this.GetValidationSampler(validation)
        );
        if (useExistingWeights)
        {
            // After initialization of network (trainer)
            // Restore trained weights
            RestoreWeights(network);
        }

        Console.WriteLine("Info:");
        var networkName = network is ArchitectureBlock nameArch ? nameArch.Name : network.GetType().Name;
        Console.WriteLine($"  Example: {this.Name}");
        Console.WriteLine($"  Network: {networkName}");
        Console.WriteLine($"  Training size: {training.InputShape} x {training.Count}");
        Console.WriteLine($"  Validation size: {validation.InputShape} x {validation.Count}");
        Console.WriteLine();

        Console.WriteLine("Training...");
        var metrics = session.Current.AllMetrics().SelectMany(
            provider => provider
                .GetType()
                .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(prop => prop.CanRead)
                .Select<PropertyInfo, (string Name, Func<object?> Getter)>(prop => (Name: prop.Name, Getter: () => prop.GetValue(provider)) ) )
            .ToList();
        Stopwatch stopwatch = Stopwatch.StartNew(); stopwatch.Stop();
        (string, Func<ModuleTrainingEnumerator.Report, object>)[] defaultFields = [
            ("Epoch", (r) => r.Epoch + 1), 
            ("Time", (r) => stopwatch.Elapsed.TotalMinutes.ToString("F4") + "min"), 
            ("AvgLoss", (r) => r.Loss.Average), 
            ("MinLoss", (r) => r.Loss.Min), 
            ("MaxLoss", (r) => r.Loss.Max)
        ];
        object?[] row = new object?[defaultFields.Length + metrics.Count];
        for (var i = 0; i < defaultFields.Length; i++)
        {
            row[i] = defaultFields[i].Item1;
        }
        for (var i = 0; i < metrics.Count; i++)
        {
            row[defaultFields.Length + i] = metrics[i].Name;
        }
        using StreamWriter? log = useLogging ? CreateLogger(exampleName + ".log.csv") : null;
        if (log is not null)
        {
            foreach (var obj in row)
            {
                log.Write(obj); log.Write(", ");
            }
            log.WriteLine();
            log.Flush();
        }

        WriteRow(row);
        stopwatch.Restart();
        IProgress<ModuleTrainingEnumerator.EpochProgress> progress = new Progress<ModuleTrainingEnumerator.EpochProgress>(report =>
        {
            Console.Title = $"DotML Train - Epoch {report.Epoch + 1} {report.CompletedPercent * 100:F2}% - {report.ProcessIndex}/{report.ProcessSteps} {(report.IsTraining ? "training" : "validating")}";
        });
        while (session.MoveNext(progress))
        {
            stopwatch.Stop();
            var time = stopwatch.Elapsed;

            // Write "shared" metrics
            var report = session.Current;
            for (var i = 0; i < defaultFields.Length; i++)
            {
                row[i] = defaultFields[i].Item2(report);
            }
            if (log is not null)
            {
                for (var i = 0; i < defaultFields.Length; i++)
                {
                    log.Write(defaultFields[i].Item2(report)); log.Write(", ");
                }
            }

            // Write the "user defined" metrics
            for (var i = 0; i < metrics.Count; i++)
            {
                var value = metrics[i].Getter();
                row[defaultFields.Length + i] = value;
                if (log is not null)
                {
                    log.Write(value); log.Write(", ");
                }
            }
            if (log is not null) {
                log.WriteLine();
                log.Flush();
            }
            WriteRow(row);

            // Save weights if we want
            if (saveInterval.HasValue && saveInterval.Value > 0 && report.Epoch != 0 && report.Epoch % saveInterval.Value == 0)
            {
                if (HasTrainingProgressed(session.Current)) {
                    SaveWeights(network);
                    Console.WriteLine("Saved weights");
                } else
                {
                    Console.WriteLine("Skipped saving weights, not better than previous");
                }
            }

            OnTrainingIteration(network, training, validation, session.Current);

            stopwatch.Restart();
        }

        // Create final training reports
        if (useLogging)
        {
            foreach (var report in this.GenerateTrainingReports(network, training, validation, session.Current))
            {
                using var reportWriter = CreateLogger(exampleName + ".train." + report.Name + report.Extension);
                report.Emit(reportWriter);
            }
        }

        // Save weights
        if (HasTrainingProgressed(session.Current)) {
            SaveWeights(network);
            Console.WriteLine("Saved weights");
        } else
        {
            Console.WriteLine("Skipped saving weights, not better than previous");
        }
    }

    private float? progress = null;
    protected virtual bool HasTrainingProgressed(ModuleTrainingEnumerator.Report trainingReport)
    {
        // Save if loss is smaller than the new loss (or first save)
        // Want smaller loss
        if (!progress.HasValue || progress.Value > trainingReport.Loss.Average)
        {
            progress = trainingReport.Loss.Average;
            return true;
        }
        return false;
    }
    protected void ResetTrainingProgress() => progress = null;

    public override void Validate(bool useLogging)
    {
         var exampleName = this.GetType().Name;

        // Load network
        var network = this.GetArchitecture();
        RestoreWeights(network);

        // Load tensors (ensure that we have enough data)
        this.LoadTrainingData(out var training, out var validation);
        if (training.Count == 0)
        {
            throw new InvalidOperationException("Training data is not provided");
        }
        if (validation.Count == 0)
        {
            validation = training;
        }

        // Setup trainer (sensible defaults)
        var trainer = new ModuleTrainer();

        this.ConfigureTrainer(trainer); // Example specific configs

        Console.WriteLine("Info:");
        var networkName = network is ArchitectureBlock nameArch ? nameArch.Name : network.GetType().Name;
        Console.WriteLine($"  Example: {this.Name}");
        Console.WriteLine($"  Network: {networkName}");
        Console.WriteLine($"  Validation size: {validation.InputShape} x {validation.Count}");
        Console.WriteLine();

        Console.WriteLine("Validating...");
        var metrics = trainer.Metrics.SelectMany(
            provider => provider
                .GetType()
                .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(prop => prop.CanRead)
                .Select<PropertyInfo, (string Name, Func<object?> Getter)>(prop => (Name: prop.Name, Getter: () => prop.GetValue(provider)) ) )
            .ToList();
        Stopwatch stopwatch = Stopwatch.StartNew(); stopwatch.Stop();
        (string, Func<ModuleTrainingEnumerator.Report, object>)[] defaultFields = [
            ("Epoch", (r) => r.Epoch + 1), 
            ("Time", (r) => stopwatch.Elapsed.TotalMinutes.ToString("F4") + "min"), 
            ("AvgLoss", (r) => r.Loss.Average), 
            ("MinLoss", (r) => r.Loss.Min), 
            ("MaxLoss", (r) => r.Loss.Max)
        ];
        object?[] row = new object?[defaultFields.Length + metrics.Count];
        for (var i = 0; i < defaultFields.Length; i++)
        {
            row[i] = defaultFields[i].Item1;
        }
        for (var i = 0; i < metrics.Count; i++)
        {
            row[defaultFields.Length + i] = metrics[i].Name;
        }
        WriteRow(row);

        stopwatch.Restart();
        var report = ModuleTrainingEnumerator.Test(validation.CreateSequentialSampler(), network, trainer.Loss, 1, trainer.Metrics);
        stopwatch.Stop();

        for (var i = 0; i < defaultFields.Length; i++)
        {
            row[i] = defaultFields[i].Item2(report);
        }

        // Write the "user defined" metrics
        for (var i = 0; i < metrics.Count; i++)
        {
            var value = metrics[i].Getter();
            row[defaultFields.Length + i] = value;
        }
        WriteRow(row);

        // Create final validation reports
        if (useLogging)
        {
            foreach (var genReport in this.GenerateTrainingReports(network, validation, validation /* could use training here... */, report))
            {
                using var reportWriter = CreateLogger(exampleName + ".test." + genReport.Name + genReport.Extension);
                genReport.Emit(reportWriter);
            }
        }
    }

    public override void Clean()
    {
        if (Directory.Exists(ProcessedDataPath))
        {
            foreach (var file in Directory.EnumerateFiles(ProcessedDataPath))
            {
                File.Delete(file);
            }
        }
    }

    protected virtual void OnTrainingIteration(INetworkModule network, ITrainingDataSource<float> training, ITrainingDataSource<float> validation, ModuleTrainingEnumerator.Report report) {}

    public abstract INetworkModule GetArchitecture();
    public override bool HasBeenTrained() => File.Exists(Path.Combine(ExamplePath, DefaultWeightsFilename));
    public virtual Safetensors LoadWeights()
    {
        return Safetensors.ReadFromFile(Path.Combine(ExamplePath, DefaultWeightsFilename));
    }
    public virtual void SaveWeights(Safetensors tensors)
    {
        // Ensure none of the weights are NaN or invalid values
        static double value2Double(object? obj)
        {
            if (obj is null)
                return double.NaN;

            return Convert.ToDouble(obj);
        }
        static bool isInvalid(double val)
        {
            return double.IsNaN(val);
        }
        foreach (var key in tensors.Keys())
        {
            if (tensors.AnyIn(key, (v) => { double val = value2Double(v); return isInvalid(val); }))
            {
                throw new ArgumentException(key, "NaN values found in model weights");
            }
        }
        // Save the weights
        tensors.WriteToFile(Path.Combine(ExamplePath, DefaultWeightsFilename));
    }

    public abstract Tensor<float> ParseUserInput(string input);

    public override void ProcessRawData() {}
    public abstract void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation);
    public virtual ITrainingDataSampler<float> GetTrainingSampler(ITrainingDataSource<float> training) => training.CreateRandomSampler(allowDuplicates: false);
    public virtual ITrainingDataSampler<float> GetValidationSampler(ITrainingDataSource<float> validation) => validation.CreateSequentialSampler();
    public abstract void ConfigureTrainer(ModuleTrainer trainer);
    public virtual IEnumerable<IReport> GenerateTrainingReports(INetworkModule network, ITrainingDataSource<float> training, ITrainingDataSource<float> validation, ModuleTrainingEnumerator.Report report) => Enumerable.Empty<IReport>();
    public abstract string FormatOutput(string inputStr, Tensor<float> input, Tensor<float> output);
}