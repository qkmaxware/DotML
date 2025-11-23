using System.Text.RegularExpressions;
using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.Templates;
using DotML.Network.Training;
using SkiaSharp;

namespace DotML.Examples.Fruits;

public class Fruits : BackpropExample
{
    private const int ImgWidth = 224;
    private const int ImgHeight = 224;
    private const int ImgChannels = 3;
    private static string[] Classes = [
        "Apple",
        "Avocado",
        "Banana",
        "Cherry",
        "Grape",
        "Orange",
        "Peach",
        "Pineapple",
        "Strawberry",
        "Watermelon"
    ];

    public override INetworkModule GetArchitecture()
    {
        var factory = new VGGFactory();

        // Create a network which operates on 32x32 pixel colour images

        var settings = VGGFactory.BuildSettings.VGG7();
        settings.ActivationFunction = ActivationFunctions.LeakyReLU;
        settings.ImageHeight = ImgHeight;
        settings.ImageWidth = ImgWidth;
        settings.ImageChannels = ImgChannels;
        settings.OutputClasses = Classes.Length;
        settings.UseBatchNorm = false;

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
        var tensor = scaled.ToColourTensor();
        tensor.ElementWiseInplace((x) => x / 255.0f);
        return tensor;
    }

    public override void ProcessRawData()
    {
        // Find all files
        var files = new DirectoryInfo(RawDataPath).EnumerateFiles("*.*", SearchOption.AllDirectories)
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
        augmentor.SelectionPercent = Distributions.Uniform<float>(0.6, 0.9);
        augmentor.Brightness = Distributions.Uniform<float>(0.9f, 1.1f);
        augmentor.Contrast = Distributions.Uniform<float>(0.9f, 1.1f);
        augmentor.AllowHorizontalFlip = true;
        augmentor.AllowVerticalFlip = false;
        augmentor.AllowInverting = false;

        // Foreach img file
        bool firstFile = true;
        foreach (var file in files)
        {
            // Load image
            using var bitmap = SKBitmap.Decode(file.FullName);
            var baseName = Path.GetFileNameWithoutExtension(file.Name);

            int classIndex = -1;
            for (var i = 0; i < Classes.Length; i++) {
                if ((file.DirectoryName ?? string.Empty).Contains(Classes[i], StringComparison.CurrentCultureIgnoreCase))
                {
                    classIndex = i;
                    break;
                }
            }
            if (classIndex == -1) {
                continue;
            }

            // Perform augmentations
            using var augments = augmentor.Augment(bitmap, augmentations: 20);

            int augmentIndex = 0;
            foreach (var augment in augments)
            {
                if (firstFile) {
                    var pngName = Path.Combine(ProcessedDataPath, baseName + "." + augmentIndex + ".class" + classIndex + ".png");
                    using var pngStream = File.Open(pngName, FileMode.Create);
                    augment.Encode(pngStream, SKEncodedImageFormat.Png, 100);
                }

                var tensor = augment.ToColourTensor();
                var tensorName = Path.Combine(ProcessedDataPath, baseName + "." + augmentIndex + ".class" + classIndex + ".json");
                using var writer = new StreamWriter(tensorName);
                tensor.SaveJson(writer);

                augmentIndex++;
            }

            firstFile = false;
        }
    }

    public override void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation)
    {
        var rng = Random.Shared;

        var ishape = new TensorShape(ImgChannels, ImgHeight, ImgWidth);
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

        var classExtract = new Regex(@"\.class(?<class>[0-9]+)", RegexOptions.Compiled);

        foreach (var file in Directory.GetFiles(ProcessedDataPath, "*.json", SearchOption.TopDirectoryOnly))
        {
            // Fetch one-hot coded ouput vector (saved in filename)
            var match = classExtract.Match(file);
            if (!match.Success) {
                continue;
            }

            var classIndex = int.Parse(match.Groups["class"].Value);
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
                trn.Add((input, output));
                val.Add((input.Clone(), output.Clone()));
            }
        }

        training = trn;
        validation = val;
    }

    public override void ConfigureTrainer(ModuleTrainer trainer)
    {
        trainer.MaxEpochs = 50;
        trainer.LearningRateScheduler = new RampUpWarmup(
            maxWarmupRate: 1e-3f,
            warmupEpochs: 5,
            scheduler: new CosineAnnealing(1e-3f, 250)
        );
        trainer.BatchSize = 16;
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

    protected override void OnTrainingIteration(INetworkModule network, ITrainingDataSource<float> training, ITrainingDataSource<float> validation, ModuleTrainingEnumerator.Report report)
    {
        // For debugging
        var reportA = new ConfusionMatrixReport("validation-confusion", Classes, report.Metrics<AccuracyMetricsProvider>().GetConfusionMatrix());
        using (var writer = new StreamWriter(report.Epoch + "validation-confusion.shared.csv"))
            reportA.Emit(writer);
        
        var validDataReport = ModuleTrainingEnumerator.Test(validation.CreateSequentialSampler(), network, LossFunctions.CategoricalCrossEntropy, 1, new AccuracyMetricsProvider());
        var reportC = new ConfusionMatrixReport("validation-confusion", Classes, validDataReport.Metrics<AccuracyMetricsProvider>().GetConfusionMatrix());
        using (var writer = new StreamWriter(report.Epoch + "validation-confusion.new.csv"))
            reportC.Emit(writer);

        var trainingDataReport = ModuleTrainingEnumerator.Test(training.CreateSequentialSampler(), network, LossFunctions.CategoricalCrossEntropy, 1, new AccuracyMetricsProvider());
        var reportB = new ConfusionMatrixReport("training-confusion", Classes, trainingDataReport.Metrics<AccuracyMetricsProvider>().GetConfusionMatrix());
        using (var writer = new StreamWriter(report.Epoch + "training-confusion.csv"))
            reportB.Emit(writer);
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