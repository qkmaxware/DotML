using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
using CommandLine;
using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.Templates;
using DotML.Network.Training;
using SkiaSharp;

namespace DotML.Examples.Binop;

public class Upscale : BackpropExample
{
    public int UpscaleFactor = 2;
    const float Coverage  = 0.4f;
    const int LowResTileWidth = 17;
    const int LowResTileHeight = 17;
    int HighResTileWidth => LowResTileWidth * UpscaleFactor;
    int HighResTileHeight => LowResTileHeight * UpscaleFactor;

    public class ExtraArguments
    {
        // User can choose the scaling factor they want (within limit and a power of 2)
        public int Factor { get; set; }
    }
    public override void Configure(string json)
    {
        if (!TryParseConfigString<ExtraArguments>(json, out var args))
        {
            throw new FormatException("Failed to parse configuration json");
        }

        // Ensure its divisible by 2
        if (args.Factor % 2 != 0)
            args.Factor = args.Factor - 1; 

        this.UpscaleFactor = Math.Clamp(args.Factor, 2, 8);
    }

    public override string? GetDescription() => $"Upscaling of images by a configurable factor (at least 2x).";

    public override INetworkModule GetArchitecture()
    {
        var factory = new ESPCNFactory();

        var settings = new ESPCNFactory.BuildSettings();
        settings.ImgChannels = 1;
        settings.ImgWidth = LowResTileWidth;
        settings.ImgHeight = LowResTileHeight;
        settings.Activation = ActivationFunctions.Tanh;
        settings.UpscalingFactor = this.UpscaleFactor;

        /*var factory = new FSRCNNFactory();

        var settings = new FSRCNNFactory.BuildSettings();
        settings.ImgChannels = 1;
        settings.ImgWidth = LowResTileWidth;
        settings.ImgHeight = LowResTileHeight;
        settings.Activation = ActivationFunctions.LeakyReLU;
        settings.UpscalingFactor = this.UpscaleFactor;*/

        return factory.Make(settings);
    }

    public override Safetensors LoadWeights()
    {
        // IE Xor.Network.safetensors or And.Network.safetensors
        // Allows for separate weights per operation
        return Safetensors.ReadFromFile(Path.Combine(ExamplePath, UpscaleFactor + "x." + DefaultWeightsFilename));
    }

    public override void SaveWeights(Safetensors tensors)
    {
        // IE Xor.Network.safetensors or And.Network.safetensors
        // Allows for separate weights per operation
        tensors.WriteToFile(Path.Combine(ExamplePath, UpscaleFactor + "x." + DefaultWeightsFilename));
    }

    public override Tensor<float> ParseUserInput(string input)
    {
        // Load image (1080 x 1618)
        using var bitmap = SKBitmap.Decode(input);
        
        // Convert to tensor
        var luminance = bitmap.ToYCrCb();
        var tensor = luminance.ToTensor((yCrCb) => yCrCb.Y / 255.0f); // 0-1

        using var grey = tensor.ToGreyscaleBitmaps();
        using var greyStream = File.Open(Path.GetFileNameWithoutExtension(input) + ".greyscale.png", FileMode.Create);
        grey[0].Encode(greyStream, SKEncodedImageFormat.Png, 100);
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

    var rng = Random.Shared;

    // Foreach img file
    using var iWriter = new BinaryWriter(File.Open(Path.Combine(ProcessedDataPath, "inputs.bin"), FileMode.Create));
    using var oWriter = new BinaryWriter(File.Open(Path.Combine(ProcessedDataPath, "outputs.bin"), FileMode.Create));

    int fileIndex = 0;
    foreach (var file in files)
    {
        // Load image
        using var bitmap = SKBitmap.Decode(file.FullName);
        var baseName = Path.GetFileNameWithoutExtension(file.Name);
        var width = bitmap.Width;
        var height = bitmap.Height;

        if (width < HighResTileWidth || height < HighResTileHeight)
            continue;

        // compute tile size (fall back to image size if smaller)
        int tileW = HighResTileWidth;
        int tileH = HighResTileHeight;

            // total area and desired covered area
        long totalArea = (long)width * height;
        long tileArea = (long)tileW * tileH;
        if (tileArea <= 0 || totalArea <= 0)
            continue;

        // number of tiles needed to reach coverage (round up)
        long desiredCovered = (long)Math.Ceiling(Coverage * totalArea);
        int numTiles = (int)Math.Max(1, Math.Ceiling((double)desiredCovered / tileArea));

        int maxOffsetX = Math.Max(0, width - tileW);
        int maxOffsetY = Math.Max(0, height - tileH);

        for (int i = 0; i < numTiles; i++)
        {
            int offsetX = maxOffsetX > 0 ? rng.Next(0, maxOffsetX + 1) : 0;
            int offsetY = maxOffsetY > 0 ? rng.Next(0, maxOffsetY + 1) : 0;

            var rect = new SKRectI(offsetX, offsetY, offsetX + tileW, offsetY + tileH);
            using var tileBitmap = new SKBitmap(tileW, tileH);
            bitmap.ExtractSubset(tileBitmap, rect);

            CreateHrLrPair(baseName, fileIndex, i, 0, tileBitmap, iWriter, oWriter);
        }

        fileIndex++;
    }
}
    private void CreateHrLrPair(string baseName, int fileIndex, int sliceIndex, int augmentIndex, SKBitmap tileBitmap, BinaryWriter iWriter, BinaryWriter oWriter)
    {
        // HR
        {
            var luminance = tileBitmap.ToYCrCb();

            // PNG for debugging
            if (fileIndex == 0 && sliceIndex < 10) {
                var tensor = luminance.ToTensor((y) => y.Y);
                var pngName = Path.Combine(ProcessedDataPath, baseName + ".slice" + sliceIndex + ".augment" + augmentIndex + ".HR.png");
                WriteLuminancePng(pngName, tensor);
            }

            //var tensorName = Path.Combine(ProcessedDataPath, baseName + ".slice" + sliceIndex+ ".augment" + augmentIndex + ".HR.json");
            //using var writer = new StreamWriter(tensorName);
            //tensor.SaveJson(writer);
            for (var r = 0; r < luminance.GetLength(0); r++)
            {
                for (var c = 0; c < luminance.GetLength(1); c++)
                {
                    oWriter.Write(luminance[r, c].Y);
                }
            }
        }
        // LR
        {
            using var scaledTileBitmap = tileBitmap.Resize(new SKSizeI(tileBitmap.Width / UpscaleFactor, tileBitmap.Height / UpscaleFactor), SKSamplingOptions.Default);
            var luminance = scaledTileBitmap.ToYCrCb();

            // PNG for debugging
            if (fileIndex == 0 && sliceIndex < 10) {
                var tensor = luminance.ToTensor((y) => y.Y);
                var pngName = Path.Combine(ProcessedDataPath, baseName + ".slice" + sliceIndex + ".augment" + augmentIndex + ".LR.png");
                WriteLuminancePng(pngName, tensor);
            }

            //var tensorName = Path.Combine(ProcessedDataPath, baseName + ".slice" + sliceIndex + ".augment" + augmentIndex + ".LR.json");
            //using var writer = new StreamWriter(tensorName);
            //tensor.SaveJson(writer);
            for (var r = 0; r < luminance.GetLength(0); r++)
            {
                for (var c = 0; c < luminance.GetLength(1); c++)
                {
                    iWriter.Write(luminance[r, c].Y);
                }
            }
        }
    }
    private static void WriteLuminancePng(string name, Tensor<float> tensor)
    {
        using var bitmap = new SKBitmap(tensor.Shape[NCHW.Columns], tensor.Shape[NCHW.Rows]);

        for (var r = 0; r < bitmap.Height; r++)
        {
            for (var c = 0; c < bitmap.Width; c++)
            {
                var luminance = (byte)tensor[0, r, c];
                bitmap.SetPixel(c, r, new SKColor(luminance, luminance, luminance));
            }
        }

        using var pngStream = File.Open(name, FileMode.Create);
        bitmap.Encode(pngStream, SKEncodedImageFormat.Png, 100);
    }

    public override void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation)
    {
        var rng = Random.Shared;

        var ishape = new TensorShape(1, LowResTileHeight, LowResTileWidth);
        var iElements = ishape.LogicalElementCount();
        var oshape = new TensorShape(1, HighResTileHeight, HighResTileWidth);
        var oElements = oshape.LogicalElementCount();
        ListTrainingDataSource<float> trn = new ListTrainingDataSource<float>(ishape, oshape);
        ListTrainingDataSource<float> val = new ListTrainingDataSource<float>(ishape, oshape);

        using var iReader = new BinaryReader(File.Open(Path.Combine(ProcessedDataPath, "inputs.bin"), FileMode.Open));
        using var oReader = new BinaryReader(File.Open(Path.Combine(ProcessedDataPath, "outputs.bin"), FileMode.Open));

        while (iReader.BaseStream.Position != iReader.BaseStream.Length)
        {
            Tensor<float> i = Tensor<float>.Defaults(ishape);
            var ispan = i.AsSpan();
            for (var j = 0; j < ispan.Length; j++)
            {
                ispan[j] = iReader.ReadByte() / 255.0f;
            }

            Tensor<float> o = Tensor<float>.Defaults(oshape);
            var ospan = o.AsSpan();
            for (var j = 0; j < ospan.Length; j++)
            {
                ospan[j] = oReader.ReadByte() / 255.0f;
            }

            if (rng.NextDouble() > 0.25)
            {
                trn.Add((i, o));
            }
            else
            {
                val.Add((i, o));
            }
        }

        training = trn;
        validation = val;
    }

    public override void ConfigureTrainer(ModuleTrainer trainer)
    {
        trainer.MaxEpochs = 250;
        trainer.LearningRateScheduler = new RampUpWarmup(
            maxWarmupRate: 1e-3f,
            warmupEpochs: 5,
            scheduler: new CosineAnnealing(1e-3f, trainer.MaxEpochs - 5)
        );
        trainer.BatchSize = 16;
        trainer.Initializer = new NormalXavierInitialization();
        trainer.Loss = LossFunctions.MeanSquaredError;
        trainer.Optimizer = new Adam();
        trainer.GlobalClipping = new GlobalMagnitudeClipping<float>(10);// Not required, but safe
        trainer.LocalClipping = null;
        trainer.Regularization = new NoRegularization(); //new L2Regularization(1e-4f);
        trainer.Patience = 3; // Patience here relates only to the stop condition below
        trainer.StopCondition = static (report) => report.Epoch > 20 && report.Metrics<SignalToNoiseProvider>().DecibelSNR.Min > 30;
        trainer.Metrics.Add(new SignalToNoiseProvider(1.0f));
        trainer.Metrics.Add(new StructuralSimilarityIndexProvider(1.0f));
    }

    public override string FormatOutput(string inputStr, Tensor<float> input, Tensor<float> output)
    {
        var basename = Path.GetFileNameWithoutExtension(inputStr);
        var tensorName = basename + "." + UpscaleFactor + "x.tensor.json";
        var pngName = basename + "." + UpscaleFactor + "x.png";

        // Save tensor (debug)
        Metric<float> f = new Metric<float>();
        foreach (var e in output.EnumerateElements())
            f.AddSample(e);
        Console.WriteLine($"min: {f.Min}, max: {f.Max}, avg: {f.Average}, std: {f.StandardDeviation}, var: {f.Variance}");
        using var tensorWriter = new StreamWriter(tensorName);
        output.SaveJson(tensorWriter);

        // Convert back to an image
        using var bitmaps = output.ToGreyscaleBitmaps();
        var bitmap = bitmaps[0];
        var names = new List<string>();
        
        // Save image
        using var stream = File.Open(pngName, FileMode.Create);
        bitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
        names.Add(pngName);

        // TODO re-add colour into the image via bilinear interpolation 

        return string.Join(Environment.NewLine, names.Select(name => $"Upscale saved: '{name}'"));
    }

}