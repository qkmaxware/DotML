using DotML;
using DotML.Network;
using DotML.Network.Training;
using DotML.Network.Initialization;
using System.Reflection;
using System.Diagnostics;
using DotML.Network.IO;

public class Logger {
    private TextWriter writer;
    public Logger(TextWriter writer) {
        this.writer = writer;
    } 
    public void Write(object? o) {
        var s = o?.ToString() ?? string.Empty;
        writer.Write(s); writer.Flush();
        Console.Write(s);
    }
    public void WriteLine() {
        writer.WriteLine(); writer.Flush();
        Console.WriteLine();
    }
    public void WriteLine(object? o) {
        var s = o?.ToString() ?? string.Empty;
        writer.WriteLine(s); writer.Flush();
        Console.WriteLine(s);
    }
}
public class Program {

public static void Main() {
    #region Network
    var output_labels = new string[]{ "Apple", "Orange" };      // Labels for each output class
    var network = AlexNet.Make(
        AlexNet.Version.V1,                                     // AlexNet architecture version to copy
        output_classes: output_labels.Length,                   // Number of output classes
        img_channels: 3, img_width: 224, img_height: 224,       // Image size, 3 channels is RGB
        activation: ActivationFunctions.ReLU
    );
    network.InsertLayerBefore((after) => new LayerNorm(after.InputShape), (index, layer) => layer is PoolingLayer);
    network.InsertLayerAfter((after) => new LayerNorm(after.OutputShape), (index, layer) => layer is DenseLinearLayer);
    using var log_writer = new StreamWriter($"{network.Name}.log");
    var logger = new Logger(log_writer);
    using (var writer = new StreamWriter($"{network.Name}.netbuild")) {
        var netbuilder = new NetbuildSerializer();
        netbuilder.Serialize(network, writer);
    }
    logger.WriteLine($"Created network: {network.Name} (layers: {network.LayerCount}, shape: {network.InputShape} -> {network.OutputShape})");
    logger.WriteLine();
    #endregion

    #region Trainer
    var validation_report = new DefaultValidationReport();
    var performance_report = new DefaultProfilingReport();
    var trainer = new EnumerableBatchTrainer<FeedforwardNetwork> {
        Epochs = 250,                                           // Max epochs to train for
        LearningRate = 0.001,                                   // Default weight adjustment rate
        LearningRateOptimizer = Optimizers.Adam,                // Weight update optimizer
        LossFunction = LossFunctions.CategoricalCrossEntropy,   // Loss function 
        NetworkInitializer = Initializers.He,                   // Initialization method
        Regularization = Regularization.None,                   // Regularization method
        BatchSize = 8,                                          // Batches to execute in parallel (CPU logical cores)
        EnableGradientClipping = false,                         // Check if gradient clipping should be used
        ClippingThresholdSynapses = 10,                         // If gradient clipping, clip weights to this value
        ClippingThresholdBiases = 5.0,                          // If gradient clipping, clip biases to this value
        EarlyStop = true,                                       // Check if the trainer can stop before max epochs reached
        EarlyStopAccuracy = 0.1,                                // Desired loss at which early stop can be triggered
        EarlyStopPatience = 1,                                  // Number of epochs where early stop condition is met before training is halted
        ValidationReport = validation_report,
        Profiler = performance_report
    };
    logger.WriteLine($"Trainer configured: {trainer.GetType().Name} (epochs: {trainer.Epochs})");
    logger.WriteLine();
    #endregion

    #region Data
    var input_filename = "../Data/Fruits/apple-or-banana.training.bin";
    var data = ReadSerializedTrainingSet(input_filename);       // Data to use for training
    var validation = data;                                      // Data to use for validation and early stop
    var first = data[0];
    logger.WriteLine($"Training data loaded: {input_filename} (items: {data.Size}, input size: {first.Input.Dimensionality}, output size: {first.Output.Dimensionality})");
    if (network.InputShape.Count != first.Input.Dimensionality) {
        logger.Write("    ");
        logger.Write($"Input shape mismatch between network input {network.InputShape} ({network.InputShape.Count}) and training input size {first.Input.Dimensionality}");
        logger.WriteLine();
    }
    logger.WriteLine();
    #endregion

    logger.Write("Begin training (y/n)? "); 
    var confirm = Console.ReadLine()?.ToLower() switch {
        "y" => true,
        "yes" => true,
        "true" => true,
        _ => false
    };
    if (!confirm)
        return;

    #region Training Steps
    var session = (BatchTrainerEnumerator<FeedforwardNetwork>)trainer.EnumerateTraining(
        network,                                                // Network to train
        data.SampleRandomly(),                                  // Sample the training data in no particular order
        validation.SampleSequentially()                         // Sample the validation data sequentially
    );
    var has_next = true;
    var total_time = Stopwatch.StartNew();
    logger.WriteLine("Training:");
    logger.Write("    ");
    logger.Write($"Epoch {session.CurrentEpoch + 0:000}... ");
    var before = Stopwatch.StartNew();
    session.ValidateStep();                                     // Validate the network before training starts
    before.Stop();
    logger.WriteLine("done (elapsed: " + before.Elapsed + ", avg loss: " + trainer.ValidationReport.AverageLoss + ")");
    session.Reset();
    double smallest_loss = double.PositiveInfinity;
    while (has_next) {                                          // Loop until training all epochs complete (or early stop)
        logger.Write("    ");
        logger.Write($"Epoch {session.CurrentEpoch + 1:000}... ");

        var timer = Stopwatch.StartNew();
        has_next = session.MoveNext();                          // Advance the training by 1 epoch
        timer.Stop();
        var elapsed = timer.Elapsed;

        if (trainer.ValidationReport.AverageLoss < smallest_loss) {
            Dump(network, "best-avg");
            smallest_loss = trainer.ValidationReport.AverageLoss;
        }

        logger.WriteLine("done (elapsed: " + elapsed + ", avg loss: " + trainer.ValidationReport.AverageLoss + ")");
    }
    total_time.Stop();
    logger.Write("    ");
    logger.WriteLine($"training done (elapsed: {total_time.Elapsed}, min loss: {validation_report.MinLoss}, max loss: {validation_report.MaxLoss}, avg loss: {validation_report.AverageLoss}, accuracy: {validation_report.Accuracy}, recall: {validation_report.Recall}, precision: {validation_report.Precision}, f1: {validation_report.F1Score})");
    logger.WriteLine();
    #endregion

    #region Save Weights
    var output_filename = Dump(network, "final");
    logger.WriteLine($"Weights saved: '{output_filename}'");
    #endregion
}

#region Utilities

private static string Dump(FeedforwardNetwork network, string name) {
    var output_filename = $"{network.Name}.{name}.safetensors"; // Desired output filename for trained weights
    var weights = network.ToSafetensor();                       // Store all weights in a safetensors file
    weights.WriteToFile(output_filename);                       // Dump safetensor file to disc
    return output_filename;
}

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