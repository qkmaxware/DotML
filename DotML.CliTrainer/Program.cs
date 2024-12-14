using DotML;
using DotML.Network;
using DotML.Network.Training;
using DotML.Network.Initialization;
using System.Reflection;
using System.Diagnostics;

public class Program {
public static void Main() {

    #region Network
    const int IMG_WIDTH = 227;
    const int IMG_HEIGHT = 227;
    const int IMG_CHANNELS = 3;
    const int OUT_CLASSES = 3;
    string[] OUT_CLASS_NAMES = new string[OUT_CLASSES] {
        "apple", "banana", "orange",
    };

    ConvolutionalFeedforwardNetwork network = new ConvolutionalFeedforwardNetwork(
        IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,
        new ConvolutionLayer        (input_size: new Shape3D(3, IMG_HEIGHT, IMG_WIDTH), padding: Padding.Valid, stride: 4, filters: ConvolutionFilter.Make(96, 3, 11)) { ActivationFunction = HyperbolicTangent.Instance },
        new LocalMaxPoolingLayer    (size: 3, stride: 2),
        new ConvolutionLayer        (input_size: new Shape3D(96, 27, 27), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(256, 96, 5)) { ActivationFunction = HyperbolicTangent.Instance },
        new LocalMaxPoolingLayer    (size: 3, stride: 2),
        new ConvolutionLayer        (input_size: new Shape3D(256, 13, 13), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(384, 256, 3)) { ActivationFunction = HyperbolicTangent.Instance },
        new ConvolutionLayer        (input_size: new Shape3D(384, 13, 13), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(384, 384, 3)) { ActivationFunction = HyperbolicTangent.Instance },
        new ConvolutionLayer        (input_size: new Shape3D(384, 13, 13), padding: Padding.Same, stride: 1, filters: ConvolutionFilter.Make(256, 384, 3)) { ActivationFunction = HyperbolicTangent.Instance },
        new LocalMaxPoolingLayer    (size: 3, stride: 2),
        new FullyConnectedLayer     (9216, 4096)  { ActivationFunction = HyperbolicTangent.Instance },
        new FullyConnectedLayer     (4096, 4096)  { ActivationFunction = HyperbolicTangent.Instance },
        new FullyConnectedLayer     (4096, OUT_CLASSES)  { ActivationFunction = HyperbolicTangent.Instance },
        new SoftmaxLayer            (OUT_CLASSES)
    );
    Console.WriteLine("Network configured: " + network.GetType().Name + " with " + network.LayerCount + " layers");
    //Console.Write("    "); Console.WriteLine("input: " + network.InputShape);
    for (var layerIndex = 0; layerIndex < network.LayerCount; layerIndex++) {
        var layer = network.GetLayer(layerIndex);
        //Console.Write("    "); Console.WriteLine("layer" + layerIndex + ": " + layer.OutputShape + " " + layer.GetType().Name);
    }
    #endregion

    #region Trainer
    DefaultValidationReport report = new DefaultValidationReport();
    BatchedConvolutionalEnumerableBackpropagationTrainer<ConvolutionalFeedforwardNetwork> trainer = new BatchedConvolutionalEnumerableBackpropagationTrainer<ConvolutionalFeedforwardNetwork> {
        LearningRate = 0.01,
        LearningRateOptimizer = new AdamOptimizer(),
        LossFunction = LossFunctions.CrossEntropy,
        NetworkInitializer = new NormalXavierInitialization(),
        BatchSize = 6,
        EnableGradientClipping = false,
        ValidationReport = report
    };
    Console.WriteLine("Trainer configured: " + trainer.GetType().Name);
    foreach (PropertyInfo property in trainer.GetType().GetProperties()) {
        Console.Write("    "); Console.Write(property.Name); Console.Write(": "); Console.WriteLine(property.CanRead ? property.GetValue(trainer, null) : "n/a");
    }
    #endregion

    #region Data
    var data = ReadData(OUT_CLASSES, 0.0, 1.0, (reader) => (reader.ReadByte() / 255.0));
    Console.WriteLine($"Training vectors loaded: \"data.bin\"");
    Console.Write("    "); Console.WriteLine($"Records: {data.Size}");
    Console.Write("    "); Console.WriteLine($"InputSize: {data[0].Input.Dimensionality}");
    Console.Write("    "); Console.WriteLine($"OutputSize: {data[0].Output.Dimensionality}");
    #endregion

    #region Training Steps
    var position = Console.GetCursorPosition();
    var session = trainer.EnumerateTraining(network, data.SampleRandomly(), data.SampleSequentially());
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
        Console.Write(" batches");
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
        Console.Write(data.Size);
        Console.Write(" validated");
    };
    session.OnValidated += (epoch, maxEpoch, index, accuracy) => {
        Console.SetCursorPosition(position.Left, position.Top);
        Console.Write('|');
        var percent = (float)index/(float)data.Size;
        for (float i = 0; i <= 1.0; i += progress_bar_step) {
            if (i <= percent)
                Console.Write('-');
            else
                Console.Write(' ');
        }
        Console.Write('|');
        Console.Write(index + 1);
        Console.Write('/');
        Console.Write(data.Size);
        Console.Write(" validated");
    };
    
    double? min_loss = null;
    var has_next = true;
    var reset_colour = Console.ForegroundColor;
    Console.WriteLine();
    var filename_root = $"{DateTime.Now.ToString("yyyy-dd-M--HH-mm-ss")}.";
    var filename = $"{filename_root}training-report.csv";
    Console.WriteLine($"Training started: \"{filename}\"");
    using var report_writer = new StreamWriter(filename);
    report_writer.WriteLine("epoch, min-loss, max-loss, average-loss, tests-passed, tests-failed");
    report_writer.Flush();
    while (has_next) {
        Console.Write("    ");
        Console.Write($"Epoch{session.CurrentEpoch:000}: ");
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

        // Last report
        Console.SetCursorPosition(position.Left, position.Top);
        if (min_loss.HasValue) {
            if (report.MaxLoss < min_loss.Value) {
                Console.ForegroundColor = ConsoleColor.Green;
                min_loss = report.MaxLoss;
            } else {
                Console.ForegroundColor = ConsoleColor.Red;
            }
        }
        Console.Write($"{report.TestsPassedCount}/{report.TestCount} passed, {elapsed} elapsed, {report.AverageLoss} loss,                              ");
        report_writer.WriteLine($"{session.CurrentEpoch}, {report.MinLoss}, {report.MaxLoss}, {report.AverageLoss}, {report.TestsPassedCount}, {report.TestsFailedCount}");
        report_writer.Flush();
        Console.ForegroundColor = reset_colour;
        Console.WriteLine();
    }
    #endregion

    #region Save
    var weights = network.ToSafetensor();
    weights.WriteToFile($"{filename_root}weights.safetensors");
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

private static TrainingSet ReadData(int category_count, double category_off, double category_on, Func<BinaryReader, double> element_parser) {
    using var stream = File.OpenRead("data.bin");
    using var reader = new BinaryReader(stream);
                        
    List<TrainingPair> pairs = new List<TrainingPair>();
    while (stream.Position < stream.Length) {
        var category_index  = reader.ReadByte();
        var vector_size     = reader.ReadInt32();
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