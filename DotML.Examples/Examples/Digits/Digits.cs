using System.Text.RegularExpressions;
using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.Templates;
using DotML.Network.Training;
using SkiaSharp;

namespace DotML.Examples.Digits;

public class Digits : BackpropExample
{
    private const int ImgWidth = 32;
    private const int ImgHeight = 32;
    private static string[] Classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

    public override string? GetDescription() => $"Classification of {ImgWidth}x{ImgHeight} images as single digits 0-9.";

    public override INetworkModule GetArchitecture()
    {
        var factory = new LeNetFactory();

        // Create a LeNet network which operates on 32x32 pixel greyscale images

        var settings = new LeNetFactory.BuildSettingsV5();
        settings.Activation = ActivationFunctions.LeakyReLU;
        settings.ImgHeight = ImgHeight;
        settings.ImgWidth = ImgWidth;
        settings.ImgChannels = 1;
        settings.OutputClasses = Classes.Length;

        return factory.Make(settings);
    }

    public override Tensor<float> ParseUserInput(string input)
    {
        // Load image
        using var bitmap = SKBitmap.Decode(input);

        // Scale to desired size
        using var scaled = new SKBitmap(width: ImgWidth, height: ImgHeight, isOpaque: true);
        bitmap.ScalePixels(scaled, SKSamplingOptions.Default);

        // Convert to tensor
        var tensor = scaled.ToGreyscaleTensor();
        tensor.ElementWiseInplace((x) => x / 255.0f);
        return tensor;
    }

    public override void ProcessRawData()
    {
        // Find all files
        var files = new DirectoryInfo(RawDataPath).EnumerateFiles()
            .Where(file => file.Extension switch
            {
                ".png" => true,
                ".jpg" => true,
                ".jpeg" => true,
                ".bmp" => true,
                ".webp" => true,
                _ => false
            });

        // Configure augmentations
        var augmentor = new SKAugmentor();
        augmentor.ForcedOutputDimensions = (Width: ImgWidth, Height: ImgHeight);
        augmentor.RotationDegrees = Distributions.Uniform<float>(-15.0f, 15.0f);
        augmentor.ScalingFactors = Distributions.Uniform<float>(0.8f, 1.2f);
        augmentor.AllowHorizontalFlip = false;
        augmentor.AllowVerticalFlip = false;
        augmentor.AllowInverting = false;

        // Foreach img file
        foreach (var file in files)
        {
            // Load image
            using var bitmap = SKBitmap.Decode(file.FullName);
            var baseName = Path.GetFileNameWithoutExtension(file.Name);

            // Scale to desired size (320x32)
            using var scaled = new SKBitmap(width: Classes.Length * ImgWidth, height: ImgHeight, isOpaque: true);
            bitmap.ScalePixels(scaled, SKSamplingOptions.Default);

            // Get all digits
            using var digits = scaled.Slice(rows: 1, columns: Classes.Length);

            for (var digitIndex = 0; digitIndex < Classes.Length; digitIndex++)
            {
                var digit = digits[digitIndex];
                var classLabel = Classes[digitIndex];

                // Perform augmentations
                using var augments = augmentor.Augment(digit, augmentations: 20);

                int augmentIndex = 0;
                foreach (var augment in augments)
                {
                    var pngName = Path.Combine(ProcessedDataPath, baseName + "." + augmentIndex + ".class" + classLabel + ".png");
                    using var pngStream = File.Open(pngName, FileMode.Create);
                    augment.Encode(pngStream, SKEncodedImageFormat.Png, 100);

                    var tensor = augment.ToGreyscaleTensor();
                    var tensorName = Path.Combine(ProcessedDataPath, baseName + "." + augmentIndex + ".class" + classLabel + ".json");
                    using var writer = new StreamWriter(tensorName);
                    tensor.SaveJson(writer);

                    augmentIndex++;
                }
            }
        }
    }

    public override void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation)
    {
        var rng = Random.Shared;

        var ishape = new TensorShape(1, 32, 32);
        var oshape = new TensorShape(Classes.Length);
        ListTrainingDataSource<float> trn = new ListTrainingDataSource<float>(ishape, oshape);
        ListTrainingDataSource<float> val = new ListTrainingDataSource<float>(ishape, oshape);

        var oneHot = new Tensor<float>[Classes.Length];
        for (var i = 0; i < Classes.Length; i++)
        {
            var ten = Tensor<float>.Zeros(oshape);
            ten[i] = 1.0f;
            oneHot[i] = ten;
        }

        var classExtract = new Regex(@"class(?<class>\d+)", RegexOptions.Compiled);

        foreach (var file in Directory.GetFiles(ProcessedDataPath, "*.json", SearchOption.TopDirectoryOnly))
        {
            // Fetch one-hot coded ouput vector (saved in filename)
            var match = classExtract.Match(file);
            if (!match.Success)
                continue;

            var classIndex = int.Parse(match.Groups["class"].ValueSpan);
            var output = oneHot[classIndex];

            // Read tensor values (0-255)
            using var reader = File.Open(file, FileMode.Open);
            var input = TensorExport.FromJson<float>(reader);
            input.ElementWiseInplace((x) => x / 255.0f); // Scale from 0-255 to 0-1 scale

            if (!input.Shape.Equals(ishape))
                continue;

            // Plaec in one of the two training sets
            if (rng.NextDouble() > 0.25)
            {
                trn.Add((input, output));
            }
            else
            {
                val.Add((input, output));
            }
        }

        training = trn;
        validation = val;
    }

    public override void ConfigureTrainer(ModuleTrainer trainer)
    {
        trainer.MaxEpochs = 250;
        trainer.LearningRateScheduler = new ReduceLROnPlateau(
            new RampUpWarmup(
                maxWarmupRate: 1e-4f,
                warmupEpochs: 5,
                scheduler: new ConstantRate(1e-4f)
            ),
            patience: 5,
            tolerance: 0.001f
        );
        trainer.BatchSize = 16;
        trainer.Initializer = new HeInitialization();
        trainer.Loss = LossFunctions.CategoricalCrossEntropy;
        trainer.Optimizer = new Adam();
        trainer.Patience = 3;
        trainer.GlobalClipping = new GlobalMagnitudeClipping<float>(10);
        trainer.LocalClipping = null;
        trainer.Regularization = new NoRegularization();
        trainer.StopCondition = static (report) => report.Metrics<AccuracyMetricsProvider>().Accuracy > 0.8f;
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