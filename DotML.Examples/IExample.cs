using System.Diagnostics;
using System.Reflection;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Examples;

public interface IExample
{
    public string Name { get; }
    public void Configure(string json);

    public void Run(IEnumerable<string> inputs, string? outputPath);
    public void ProcessRawData();
    public void Train(bool useExistingWeights, bool useLogging, int? saveInterval);
    public void Clean();
}

public abstract class Example : IExample
{
    public virtual string Name => GetType().Name;

    protected virtual string ExamplePath => Path.Combine(Environment.CurrentDirectory, "Examples", Name);
    protected virtual string ProcessedDataPath => Path.Combine(ExamplePath, "tensors");
    protected virtual string RawDataPath => Path.Combine(ExamplePath, "raws");

    public abstract void Clean();

    public abstract void Configure(string json);

    public virtual void ProcessRawData() {}

    public abstract void Run(IEnumerable<string> inputs, string? outputPath);

    public abstract void Train(bool useExistingWeights, bool useLogging, int? saveInterval);

    private DateTime startTime = DateTime.Now;

    protected StreamWriter CreateLogger(string name)
    {
        return new StreamWriter(startTime.ToString("yyyy-dd-M--HH-mm-ss") + "." + name);
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
            Console.WriteLine("Saved Weights");
        }
        catch (Exception)
        {
            // Re-throw if configured to
            if (throws)
                throw;
        }
    }

    public override void Run(IEnumerable<string> InputStrings, string? OutputPath)
    {
        // Load network
        var network = this.GetArchitecture();

        // Load weights (required)
        RestoreWeights(network, throws: true);

        // Parse user input
        if (InputStrings is null)
            return;

        using TextWriter pipe = !string.IsNullOrEmpty(OutputPath) ? CreateLogger("output.txt") : System.Console.Out;
        foreach (var str in InputStrings)
        {
            pipe.Write("> "); pipe.WriteLine(str);
            var input = this.ParseUserInput(str);
            var output = network.Forward(input);
            pipe.WriteLine(this.FormatOutput(str, input, output));
            pipe.WriteLine(); // Extra line between inputs
        }
    }

    public override void Train(bool useExistingWeights, bool useLogging, int? saveInterval)
    {
        var exampleName = this.GetType().Name;

        // Load network
        var network = this.GetArchitecture();

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

        // Do training loop
        var session = trainer.EnumerateTraining(
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
        Console.WriteLine($"  Name: {networkName}");
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
        var accuracy = session.Current.MetricsOrNull<AccuracyMetricsProvider>();
        stopwatch.Restart();
        while (session.MoveNext())
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
                SaveWeights(network);
            }

            OnTrainingIteration(network, training, validation, session.Current);

            stopwatch.Restart();
        }

        // Create final training reports
        if (useLogging)
        {
            foreach (var report in this.GenerateTrainingReports(network, training, validation, session.Current))
            {
                using var reportWriter = CreateLogger(exampleName + "." + report.Name + report.Extension);
                report.Emit(reportWriter);
            }
        }

        // Save weights
        SaveWeights(network);
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
    public virtual Safetensors LoadWeights()
    {
        return Safetensors.ReadFromFile(Path.Combine(ExamplePath, DefaultWeightsFilename));
    }
    public virtual void SaveWeights(Safetensors tensors)
    {
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