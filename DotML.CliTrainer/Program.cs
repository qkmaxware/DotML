using DotML;
using DotML.Network;
using DotML.Network.Training;
using DotML.Network.Initialization;
using System.Reflection;
using System.Diagnostics;

public class Program {
public static void Main() {
    var now = DateTime.Now.ToString("yyyy-dd-M--HH-mm-ss");
    var dir_root = now + ".reports";
    if (!Directory.Exists(dir_root)) {
        Directory.CreateDirectory(dir_root);
    }
    var filename_root = dir_root + Path.DirectorySeparatorChar;

    #region Network

    var network = LeNet.Make(LeNet.Version.V5, output_classes: 10, img_channels: 3, img_width: 32, img_height: 32, activation: TeLU.Instance);
    //MobileNet.Make(MobileNet.Version.V1, output_classes: 3, activation: ReLU.Instance);
    
    Console.WriteLine("Network configured: " + network.GetType().Name + " with " + network.LayerCount + " layers");
    if (network is INamedNetwork named) {
        Console.Write("    "); Console.WriteLine("architecture: " + named.Name);
    }
    Console.Write("    "); Console.WriteLine("input: " + network.InputShape);
    for (var layerIndex = 0; layerIndex < network.LayerCount; layerIndex++) {
        var layer = network.GetLayer(layerIndex);
        Console.Write("    "); Console.WriteLine("layer" + layerIndex + ": " + layer.OutputShape + " " + layer.ToString());
    }
    if (network is IMarkdownable md) {
        using (var writer = new StreamWriter($"{filename_root}network.md")) {
            writer.Write(md.ToMarkdown());
        }
    } else if (network is IJsonizable json) {
        using (var writer = new StreamWriter($"{filename_root}network.json")) {
            writer.Write(json.ToJson());
        }
    } else if (network is IHtmlable html) {
        using (var writer = new StreamWriter($"{filename_root}network.html")) {
            writer.Write(html.ToHtml());
        }
    } else if (network is IDiagrammable svg) {
        using (var writer = new StreamWriter($"{filename_root}network.svg")) {
            writer.Write(svg.ToSvg());
        }
    }
    #endregion

    #region Trainer
    var validation_report = new DefaultValidationReport();
    var performance_report = new DefaultProfilingReport();
    var trainer = new EnumerableBatchTrainer<FeedforwardNetwork> {
        Epochs = 100,
        LearningRate = 0.001,
        LearningRateOptimizer = new AdamOptimizer(),
        LossFunction = LossFunctions.CrossEntropy,
        NetworkInitializer = new HeInitialization(),
        BatchSize = 10,
        EnableGradientClipping = false,
        ClippingThresholdSynapses = 10,
        ClippingThresholdBiases = 5.0,
        ValidationReport = validation_report,
        Profiler = performance_report
    };
    Console.WriteLine("Trainer configured: " + trainer.GetType().Name);
    using (var trainer_prop_writer = new StreamWriter($"{filename_root}trainer.conf.yaml")) {
        trainer_prop_writer.WriteLine("Trainer:");
        trainer_prop_writer.Write("    "); trainer_prop_writer.Write("Type"); trainer_prop_writer.Write(": "); trainer_prop_writer.WriteLine(trainer.GetType().Name);
        foreach (PropertyInfo property in trainer.GetType().GetProperties()) {
            object? value = property.CanRead ? property.GetValue(trainer, null) : null;
            if (value is LossFunction loss)
                value = loss.Method.Name;
            else 
                value = value?.ToString() ?? "n/a";
            Console.Write("    "); Console.Write(property.Name); Console.Write(": "); Console.WriteLine(value);
            trainer_prop_writer.Write("    "); trainer_prop_writer.Write(property.Name); trainer_prop_writer.Write(": "); trainer_prop_writer.WriteLine(value);
        }
    }
    #endregion

    #region Data
    var training_data = Directory.GetFiles(Directory.GetCurrentDirectory(), "*.bin").Select(f => new FileInfo(f)).OrderByDescending(f => f.CreationTime).ToArray();
    if (training_data.Length <= 0) { 
        throw new FileNotFoundException("Training vectors");
    }
    Console.WriteLine($"Select training data?");
    for (var i = 0; i < training_data.Length; i++) {
        Console.WriteLine($"    {i}: '{training_data[i].Name}'");
    }
    Console.Write("> "); 
    var file = training_data[int.Parse(Console.ReadLine()?.ToLower() ?? "0")];
    // TODO don't make this a thing where I have to toggle between the two via code, make it based on smart file analysis. 
    //var data = ReadSerializedTrainingSet(file.FullName);
    var data = ReadClassifiedBinaryVectors(file.FullName, 10, 0.0, 1.0, (b) => b.ReadByte() / 255.0, fixed_vector_size: 1024 * 3);
    var all_data_count = (double)data.Size;
    if (data.Size == 0) 
        throw new FormatException("Empty training set");
    Console.WriteLine($"Training vectors loaded: \"{file.Name}\"");
    Console.Write("    "); Console.WriteLine($"Records: {all_data_count}");
    Console.Write("    "); Console.WriteLine($"InputSize: {data[0].Input.Dimensionality}");
    Console.Write("    "); Console.WriteLine($"OutputSize: {data[0].Output.Dimensionality}");

    var elems = data.SplitProbabilistically(3, 1).ToArray(); // new TrainingSet[]{data, data};
    data = elems[0];            // Train on 3/4 of the data
    var validation = elems[1];  // Validate against 1/4 of the data
    if (validation.Size == 0)
        validation = data;
    Console.Write("    "); Console.WriteLine($"TrainingPercent: {data.Size} ({(data.Size / all_data_count) * 100}%)");
    Console.Write("    "); Console.WriteLine($"ValidationPercent: {validation.Size} ({(validation.Size / all_data_count) * 100}%)");
    #endregion

    #region Training Steps
    var position = Console.GetCursorPosition();
    var session = trainer.EnumerateTraining(network, data.SampleRandomly(), validation.SampleSequentially());
    session.Reset();
    var checkpoints = Directory.GetFiles(Directory.GetCurrentDirectory(), "*.safetensors").Select(f => new FileInfo(f)).OrderByDescending(f => f.CreationTime).ToArray();
    if (checkpoints.Length > 0) {
        var check = checkpoints[0];
        Console.WriteLine($"Previous session weights found '{check.Name}'. Reload weights (y/n)?");
        Console.Write("> "); var read = Console.ReadLine()?.ToLower();
        switch (read) {
            case "y":
            case "yes":
            case "true":
                var tensors = Safetensors.ReadFromFile(check);
                network.FromSafetensor(tensors);
                break;
        }
    }
    const float progress_bar_step = 0.05f;
    session.OnBatchEnd += (int batch, int batchCount) => {
        Console.SetCursorPosition(position.Left, position.Top);
        Console.Write('|');
        var percent = (float)batch/(float)batchCount;
        for (float i = 0; i <= 1.0; i += progress_bar_step) {
            if (i <= percent)
                Console.Write('-');
            else
                Console.Write(' ');
        }
        Console.Write('|');
        Console.Write(batch + 1);
        Console.Write('/');
        Console.Write(batchCount);
        Console.Write(" batches trained");
    };
    session.OnValidationStart += (epoch, maxEpoch) => {
        Console.SetCursorPosition(position.Left, position.Top);
        Console.Write('|');
        for (float i = 0; i <= 1.0; i += progress_bar_step) {
                Console.Write(' ');
        }
        Console.Write('|');
        Console.Write(0);
        Console.Write('/');
        Console.Write(validation.Size);
        Console.Write(" validated      ");
    };
    session.OnValidated += (epoch, maxEpoch, index, accuracy) => {
        Console.SetCursorPosition(position.Left, position.Top);
        Console.Write('|');
        var percent = (float)index/(float)validation.Size;
        for (float i = 0; i <= 1.0; i += progress_bar_step) {
            if (i <= percent)
                Console.Write('-');
            else
                Console.Write(' ');
        }
        Console.Write('|');
        Console.Write(index + 1);
        Console.Write('/');
        Console.Write(validation.Size);
        Console.Write(" validated      ");
    };
    
    double? min_loss = null;
    var has_next = true;
    var reset_colour = Console.ForegroundColor;
    Console.WriteLine();
    var filename = $"{filename_root}training-report.csv";
    Console.WriteLine($"Training started: \"{filename}\"");
    using var report_writer = new StreamWriter(filename);
    report_writer.WriteLine("epoch, validation-min-loss, validation-max-loss, validation-average-loss, validation-accuracy, validation-precision, validation-recall, validation-f1, validation-tests-passed, validation-tests-failed, training-min-loss, training-max-loss, training-average-loss, training-accuracy, training-precision, training-recall, training-f1, training-tests-passed, training-tests-failed");
    report_writer.Flush();

    var fitness_report = new DefaultValidationReport();
    while (has_next) {
        Console.Write("    ");
        Console.Write($"Epoch{session.CurrentEpoch + 1:000}: ");
        position = Console.GetCursorPosition();

        Console.Write('|');
        for (float i = 0; i <= 1.0; i += progress_bar_step) {
                Console.Write(' ');
        }
        Console.Write('|'); 

        var timer = Stopwatch.StartNew();
        has_next = session.MoveNext();
        timer.Stop();
        var elapsed = timer.Elapsed;

        {
            var data_iterator = data.SampleSequentially(); var count = 0;
            var max_error = double.MinValue;
            var all_less_threshold = true;
            fitness_report.Reset();
            List<(FeatureSet<double> InMatrix, Vec<double> In, Vec<double> Out)> batch = new List<(FeatureSet<double> InMatrix, Vec<double> In, Vec<double> Out)>();
            var concurrency_level = trainer.BatchSize; // or Environment.ProcessorCount
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
                    
                    var loss = trainer.LossFunction(predicted, @true);
                    max_error = Math.Max(max_error, loss);
                    var passed = loss < trainer.EarlyStopAccuracy;
                    all_less_threshold &= passed;
                    fitness_report.Append(input, @true, predicted, passed, loss);
                    count ++;
                }

                // Update UI
                Console.SetCursorPosition(position.Left, position.Top);
                Console.Write('|');
                var percent = (float)(count - 1)/(float)data.Size;
                for (float i = 0; i <= 1.0; i += progress_bar_step) {
                    if (i <= percent)
                        Console.Write('-');
                    else
                        Console.Write(' ');
                }
                Console.Write('|');
                Console.Write(count);
                Console.Write('/');
                Console.Write(data.Size);
                Console.Write(" fitting checked");

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

        // Last report
        Console.SetCursorPosition(position.Left, position.Top);
        var status_char = ' ';
        if (min_loss.HasValue) {
            if (validation_report.AverageLoss < min_loss.Value) {
                Console.ForegroundColor = ConsoleColor.Green;
                min_loss = validation_report.AverageLoss;
                status_char = '+';
            } else if (validation_report.AverageLoss > min_loss.Value) {
                Console.ForegroundColor = ConsoleColor.Red;
                status_char = '-';
            }
        } else {
            min_loss = validation_report.AverageLoss;
        }
        Console.Write($"{status_char} {validation_report.TestsPassedCount}/{validation_report.TestCount} passed, {elapsed} elapsed, {validation_report.AverageLoss} validation, {fitness_report.AverageLoss} fitting, ");
        
        report_writer.WriteLine($"{session.CurrentEpoch}, {validation_report.MinLoss}, {validation_report.MaxLoss}, {validation_report.AverageLoss}, {validation_report.Accuracy}, {validation_report.Precision}, {validation_report.Recall}, {validation_report.F1Score}, {validation_report.TestsPassedCount}, {validation_report.TestsFailedCount}, {fitness_report.MinLoss}, {fitness_report.MaxLoss}, {fitness_report.AverageLoss}, {fitness_report.Accuracy}, {fitness_report.Precision}, {fitness_report.Recall}, {fitness_report.F1Score}, {fitness_report.TestsPassedCount}, {fitness_report.TestsFailedCount}");
        report_writer.Flush();
        var epochfname = $"{filename_root}epoch-{session.CurrentEpoch}.safetensors";
        network.ToSafetensor().WriteToFile(epochfname);
        Console.WriteLine($"weights 'epoch-{session.CurrentEpoch}.safetensors'");
        Console.ForegroundColor = reset_colour;

        // Emit performance metrics, always re-write and not append (unlike the validation report)
        if (performance_report is not null) {
            using (var performance_writer = new StreamWriter($"{filename_root}timings.csv")) {
                performance_writer.WriteLine("benchmark, min-time (s), max-time (s), average-time (s), total-time (s), sample-size");
                
                foreach (var metric in performance_report.Benchmarks.OrderBy(x => x.Name)) {
                    performance_writer.WriteLine($"{metric.Name}, {metric.Min.TotalSeconds}, {metric.Max.TotalSeconds}, {metric.Average.TotalSeconds}, {metric.Sum.TotalSeconds}, {metric.Count}");
                }
            }
        }
    }
    #endregion

    #region Save
    var weights = network.ToSafetensor();
    weights.WriteToFile($"{filename_root}final_weights.safetensors");
    #endregion
}

#region Utilities

private static Vec<double> VectorFromLabelIndex(int index, int classes, double off = -1, double on = 1) {
    double[] values = new double[classes];
    Array.Fill(values, off);
    if (index >= 0 && index < classes)
        values[index] = on;
    return Vec<double>.Wrap(values);
}


private static TrainingSet ReadSerializedTrainingSet(string path) {
    TrainingSet set = new TrainingSet();

    using var stream = File.OpenRead(path);
    using var reader = new BinaryReader(stream);

    set.AddFrom(reader);

    return set;
}

private static TrainingSet ReadClassifiedBinaryVectors(string path, int category_count, double category_off, double category_on, Func<BinaryReader, double> element_parser, int? fixed_vector_size = null) {
    using var stream = File.OpenRead(path);
    using var reader = new BinaryReader(stream);
                        
    List<TrainingPair> pairs = new List<TrainingPair>();
    while (stream.Position < stream.Length) {
        var category_index  = reader.ReadByte();
        var vector_size     = fixed_vector_size.HasValue ? fixed_vector_size.Value : reader.ReadInt32();
        double[] input_vec  = new double[vector_size];

        for (var i = 0; i < vector_size; i++) {
            try {
                input_vec[i] = element_parser(reader);
            } catch {
                input_vec[i] = default(double);
            }
        } 
        pairs.Add(new TrainingPair { Input = Vec<double>.Wrap(input_vec), Output = VectorFromLabelIndex(category_index, category_count, category_off, category_on) });
    }

    return new TrainingSet(pairs);
}

#endregion
}