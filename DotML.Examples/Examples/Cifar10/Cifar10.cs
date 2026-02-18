using System.Text.RegularExpressions;
using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.Templates;
using DotML.Network.Training;
using SkiaSharp;

namespace DotML.Examples.Cifar10;

public class Cifar10 : BackpropExample
{
    private const int ImgWidth = 32;
    private const int ImgHeight = 32;
    private const int ImgChannels = 3;
    private static string[] Classes = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck"
    ];

    public override string? GetDescription() => $"Classification of {ImgWidth}x{ImgHeight} images into 10 classes using the Cifar-10 training dataset.";

    public override INetworkModule GetArchitecture()
    {
        var factory = new VGGFactory();

        var settings = new VGGFactory.BuildSettings();
        settings.ImageHeight = ImgHeight;
        settings.ImageWidth = ImgWidth;
        settings.ImageChannels = ImgChannels;
        settings.ActivationFunction = ActivationFunctions.LeakyReLU;
        settings.BaseFilters = 32;
        settings.ConvolutionBlockSizes = [2, 2, 2];
        settings.HiddenNeuronCounts = [256];
        settings.OutputClasses = Classes.Length;
        settings.UseBatchNorm = false;

        var network = factory.Make(settings);
        network.ForwardShape(new TensorShape(ImgChannels, ImgHeight, ImgWidth)); // Assert that we can actually use this network
        return network;
    }

    public override void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation)
    {
        string[] training_files = [
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin",
        ];
        string[] validation_files = [
            "test_batch.bin"
        ];

        var ishape = new TensorShape(ImgChannels, ImgHeight, ImgWidth);
        var oshape = new TensorShape(Classes.Length);

        ListTrainingDataSource<float> trn = new ListTrainingDataSource<float>(ishape, oshape);
        ListTrainingDataSource<float> vld = new ListTrainingDataSource<float>(ishape, oshape);

        foreach (var file in training_files)
        {
            ParseCifar10BatchFile(Path.Combine(RawDataPath, file), trn);
        }

        foreach (var file in validation_files)
        {
            ParseCifar10BatchFile(Path.Combine(RawDataPath, file), vld);
        }

        training = new Augmented2DClassificationDataSource(trn, variations: 5, flipX: true, flipY: false, maxShift: 3, noise: null);
        validation = vld;
    }
                                    // R       G        B
    private static float[] means = [0.4914f, 0.4822f, 0.4465f];
    private static float[] stds  = [0.2023f, 0.1994f, 0.2010f];

    private void ParseCifar10BatchFile(string path, ListTrainingDataSource<float> values)
    {
        using var file = File.Open(path, FileMode.Open);
        using var reader = new BinaryReader(file);

        var oneHot = new Tensor<float>[Classes.Length];
        for (var i = 0; i < Classes.Length; i++)
        {
            var ten = Tensor<float>.Zeros(new TensorShape(Classes.Length));
            ten[i] = 1.0f;
            oneHot[i] = ten;
        }

        for (var i = 0; i < 10_000; i++)
        {
            var classIndex = reader.ReadByte();
            var input = Tensor<float>.Defaults(new TensorShape(ImgChannels, ImgHeight, ImgWidth));
            var inputSpan = input.AsSpan();
            for (int c = 0; c < ImgChannels; c++)
            {
                var mean = means[c];
                var std = stds[c];

                for (int y = 0; y < ImgHeight; y++)
                {
                    for (int x = 0; x < ImgWidth; x++)
                    {
                        float value = reader.ReadByte() / 255f;
                        input[c, y, x] = (value - mean) / std;
                    }
                }
            }
            var output = oneHot[classIndex];

            values.Add((input, output));
        }
    }

    public override Tensor<float> ParseUserInput(string input)
    {
        // Load image
        using var bitmap = SKBitmap.Decode(input);

        // Scale to desired size
        using var scaled = new SKBitmap(width: ImgWidth, height: ImgHeight, isOpaque: true);
        bitmap.ScalePixels(scaled, SKSamplingOptions.Default);

        // Convert to tensor (normalize)
        var tensor = scaled.ToColourTensor();
        for (int c = 0; c < ImgChannels; c++)
        {
            var mean = means[c];
            var std = stds[c];

            for (int y = 0; y < ImgHeight; y++)
            {
                for (int x = 0; x < ImgWidth; x++)
                {
                    float value = tensor[c, y, x] / 255f;
                    tensor[c, y, x] = (value - mean) / std;
                }
            }
        }
        return tensor;
    }

    public override void ConfigureTrainer(ModuleTrainer trainer)
    {
        trainer.MaxEpochs = 250;
        trainer.LearningRateScheduler = new RampUpWarmup(
            maxWarmupRate: 1e-3f,
            warmupEpochs: 5,
            scheduler: new CosineAnnealing(1e-3f, trainer.MaxEpochs)
        );
        trainer.BatchSize = 128;
        trainer.Initializer = new HeInitialization();
        trainer.Loss = LossFunctions.CategoricalCrossEntropy;
        trainer.Optimizer = new AdamW(weightDecay: 0.00025f);
        trainer.GlobalClipping = null;
        trainer.LocalClipping = null;
        trainer.Regularization = new NoRegularization(); // L1/L2 regularization
        trainer.Patience = 3; // Patience represents how many times the StopCondition must be met in a row
        trainer.StopCondition = static (report) => report.Metrics<AccuracyMetricsProvider>().Accuracy > 0.6f;
        trainer.Metrics.Add(new AccuracyMetricsProvider());
    }

    public override IEnumerable<IReport> GenerateTrainingReports(INetworkModule network, ITrainingDataSource<float> training, ITrainingDataSource<float> validation, ModuleTrainingEnumerator.Report report)
    {
        yield return new ConfusionMatrixReport("validation-confusion", Classes, report.Metrics<AccuracyMetricsProvider>().GetConfusionMatrix());
        var trainingDataReport = ModuleTrainingEnumerator.Test(training.CreateSequentialSampler(), network, LossFunctions.CategoricalCrossEntropy, 1, new AccuracyMetricsProvider());
        yield return new ConfusionMatrixReport("training-confusion", Classes, trainingDataReport.Metrics<AccuracyMetricsProvider>().GetConfusionMatrix());
    }

    public override string FormatOutput(string inputStr, Tensor<float> input, Tensor<float> output)
    {
        ProbabilityDistribution dist = new ProbabilityDistribution(output.AsArray(), Classes);
        return dist.ToString();
    }
}