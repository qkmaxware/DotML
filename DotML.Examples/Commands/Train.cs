using CommandLine;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;
using System.Reflection;

namespace DotML.Examples;

[Verb("train", HelpText = "Train a given example network")]
public class Train : Command
{
    [Option("raws", HelpText = "Set flag to indicate that training data is in a raw format and needs preprocessing before training")]
    public bool ProcessRaws { get; set; }

    [Option("pretrained", HelpText = "Set flag to indicate that saved weights should be used over random initialization")]
    public bool IsPretrained { get; set; }
    
    [Option("log", HelpText = "Set flag to record training logs")]
    public bool IsLogging { get; set; }

    [Option("save-every", HelpText = "Save the trained weights automatically every x epochs")]
    public int? SaveEvery {get; set;}

    public override void Exec(IExample example)
    {
        var exampleName = example.GetType().Name;

        // Load network
        var network = example.GetArchitecture();

        // Process raw data into tensors
        if (ProcessRaws)
        {
            example.ProcessRawData();
        }

        // Load tensors (ensure that we have enough data)
        example.LoadTrainingData(out var training, out var validation);
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

        example.ConfigureTrainer(trainer); // Example specific configs

        // Do training loop
        var session = trainer.EnumerateTraining(
            network: network,
            dataset: example.GetTrainingSampler(training),
            validation: example.GetValidationSampler(validation)
        );
        if (IsPretrained)
        {
            // After initialization of network (trainer)
            // Restore trained weights
            RestoreWeights(example, network);
        }

        Console.WriteLine("Info:");
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
        
        object?[] row = new object?[4 + metrics.Count];
        row[0] = "Epoch"; row[1] = "AvgLoss"; row[2] = "MinLoss"; row[3] = "MaxLoss";
        for (var i = 0; i < metrics.Count; i++)
        {
            row[4 + i] = metrics[i].Name;
        }
        using StreamWriter? log = IsLogging ? CreateLogger(exampleName + ".log.csv") : null;
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
        while (session.MoveNext())
        {
            // Write "shared" metrics
            var report = session.Current;
            row[0] = report.Epoch + 1;
            row[1] = report.Loss.Average;
            row[2] = report.Loss.Min;
            row[3] = report.Loss.Max;
            if (log is not null)
            {
                log.Write(report.Epoch + 1); log.Write(", ");
                log.Write(report.Loss.Average); log.Write(", ");
                log.Write(report.Loss.Min); log.Write(", ");
                log.Write(report.Loss.Max); log.Write(", ");
            }

            // Write the "user defined" metrics
            for (var i = 0; i < metrics.Count; i++)
            {
                var value = metrics[i].Getter();
                row[4 + i] = value;
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
            if (SaveEvery.HasValue && SaveEvery.Value > 0 && report.Epoch != 0 && report.Epoch % SaveEvery.Value == 0)
            {
                SaveWeights(example, network);
            }
        }

        // Create final training reports
        if (IsLogging)
        {
            foreach (var report in example.GenerateTrainingReports(network, training, validation, session.Current))
            {
                using var reportWriter = CreateLogger(exampleName + "." + report.Name + report.Extension);
                report.Emit(reportWriter);
            }
        }

        // Save weights
        SaveWeights(example, network);
    }
}