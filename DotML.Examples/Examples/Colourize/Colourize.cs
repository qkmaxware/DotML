using System.Text.RegularExpressions;
using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.Templates;
using DotML.Network.Training;
using SkiaSharp;

namespace DotML.Examples.Colourize;

public class Colourize : BackpropExample
{
    private const int MinImgWidth = 128;
    private const int MinImgHeight = 128;
    private const float Coverage  = 0.2f;

    public override string? GetDescription() => $"Convert black and white images of at least {MinImgWidth}x{MinImgHeight} into colour.";

    public override void ConfigureTrainer(ModuleTrainer trainer)
    {
        trainer.MaxEpochs = 250;
        trainer.LearningRateScheduler = new RampUpWarmup(
            maxWarmupRate: 1e-3f,
            warmupEpochs: 5,
            scheduler: new CosineAnnealing(1e-3f, trainer.MaxEpochs - 5)
        );
        trainer.BatchSize = 16;
        trainer.Initializer = new HeInitialization();
        trainer.Loss = LossFunctions.MeanAbsoluteError;
        trainer.Optimizer = new AdamW(weightDecay: 0.00025f);
        trainer.Patience = 3;
        trainer.GlobalClipping = new GlobalMagnitudeClipping<float>(10);
        trainer.LocalClipping = null;
        trainer.Regularization = new NoRegularization();
        trainer.StopCondition = static (report) => report.Epoch > 10 && report.Loss.Max < 0.001f;

        trainer.Metrics.Add(new SignalToNoiseProvider(maxSampleSignal: 1.0f));
        trainer.Metrics.Add(new StructuralSimilarityIndexProvider(maxValue: 1.0f));
    }

    public override Tensor<float> ParseUserInput(string inputStr)
    {
        // Load image
        using var bitmap = SKBitmap.Decode(inputStr);

        // Crop to desired aspect ratio (center-crop, like CSS "cover") then scale
        var scaled = bitmap;
        if (bitmap.Height < MinImgHeight || bitmap.Width < MinImgWidth)
        {
            // Not idea but an easy fix
            scaled = bitmap.ScaleToCover(MinImgWidth, MinImgHeight);
        }

        // Convert to greyscale if required
        var tensor = scaled.ToGreyscaleTensor();
        tensor.ElementWiseInplace(px => px / 255.0f);

        scaled.Dispose();

        return tensor;
    }

    public override string FormatOutput(string inputStr, Tensor<float> input, Tensor<float> output)
    {
        var outputGreyStr = Path.ChangeExtension(inputStr, ".greyscale.png");
        var outputColourStr = Path.ChangeExtension(inputStr, ".colourized.png");

        using var greyImgs = input.ToGreyscaleBitmaps();
        using var colourImg = output.ToColourBitmaps();

        using var greyStream = File.Open(outputGreyStr, FileMode.Create);
        greyImgs[0].Encode(greyStream, SKEncodedImageFormat.Png, 100);

        using var colourStream = File.Open(outputColourStr, FileMode.Create);
        colourImg[0].Encode(colourStream, SKEncodedImageFormat.Png, 100);

        return $"Colourized image saved to: '{outputColourStr}'";
    }

    public override INetworkModule GetArchitecture()
    {
        UNetFactory factory = new UNetFactory();
        
        var settings = new UNetFactory.BuildSettings();
        settings.Activation = ActivationFunctions.LeakyReLU;
        settings.Depth = 4;             // Input must be divisible by 16 (min 64x64)
        settings.ImageWidth = MinImgWidth;
        settings.ImageHeight = MinImgHeight;
        settings.InputChannels = 1;     // Greyscale
        settings.OutputChannels = 3;    // RGB
        settings.UseNormalization = false;

        var network = factory.Make(settings);
        var outshape = network.ForwardShape(new Shape(1, MinImgHeight, MinImgWidth));
        return network;
    }

    public override void ProcessRawData()
    {
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
        augmentor.ForcedOutputDimensions    = (Width: MinImgWidth, Height: MinImgHeight);
        augmentor.RotationDegrees           = Distributions.Uniform<float>(0, 0);
        augmentor.ScalingFactors            = Distributions.Uniform<float>(1, 1);
        augmentor.SelectionPercent          = Distributions.Uniform<float>(1, 1);       // No need, slicing makes these the right size
        augmentor.Brightness                = Distributions.Uniform<float>(0.9f, 1.1f);
        augmentor.Contrast                  = Distributions.Uniform<float>(0.9f, 1.1f);
        augmentor.AllowHorizontalFlip       = true;
        augmentor.AllowVerticalFlip         = true;
        augmentor.AllowInverting            = false;

        using var tWriter = new BinaryWriter(File.Open(Path.Combine(ProcessedDataPath, "training.bin"), FileMode.Create));
        using var vWriter = new BinaryWriter(File.Open(Path.Combine(ProcessedDataPath, "validation.bin"), FileMode.Create));

        // Loop over files
        var fileIndex = 0;
        const int tileArea = MinImgWidth * MinImgHeight;
        foreach (var file in files)
        {
            // Load image
            using var bitmap = SKBitmap.Decode(file.FullName);
            var baseName = Path.GetFileNameWithoutExtension(file.Name);

            if (bitmap.Width < MinImgWidth || bitmap.Height < MinImgHeight)
                continue;

            var totalArea = (long)bitmap.Width * (long)bitmap.Height;

            // number of tiles needed to reach coverage (round up)
            long desiredCovered = (long)Math.Ceiling(Coverage * totalArea);
            int numTiles = (int)Math.Max(1, Math.Ceiling((double)desiredCovered / tileArea));

            using var slices = bitmap.RandomSubsample(samples: 10, width: MinImgWidth, height: MinImgHeight);

            var sliceId = 0;
            foreach (var slice in slices) {

            // Perform training augmentations
            {
                using var augments = augmentor.Augment(slice, augmentations: 20);

                var augmentIndex = 0;
                foreach (SKBitmap augment in augments)
                {
                    // Create greyscale tenor and validate 
                    Tensor<float> greyTensor = augment.ToGreyscaleTensor();
                    foreach (var val in greyTensor.AsSpan()) {
                        if (val < 0 || val > 255)   
                            throw new FormatException("Invalid pixel in greyscale tensor");
                        tWriter.Write((byte)val);       // 0 - 255
                    }

                    // Create colour tensor and validate
                    Tensor<float> colourTensor = augment.ToColourTensor();
                    foreach (var val in colourTensor.AsSpan()) {
                        if (val < 0 || val > 255)   
                            throw new FormatException("Invalid pixel in colour tensor");
                        tWriter.Write((byte)val);       // 0 - 255
                    }
                    for (var y = 0; y < MinImgHeight; y++) {
                        for (var x = 0; x < MinImgWidth; x++) {
                            var pixel = augment.GetPixel(x, y);
                            var (r, g, b) = (colourTensor[0, y, x], colourTensor[1, y, x], colourTensor[2, y, x]);
                            if (pixel.Red != r || pixel.Green != g || pixel.Blue != b) {
                                throw new FormatException("Converted colour image to tensor incorrectly");
                            }
                        }
                    }

                    // Save some images (allows us to verify that the augmenting is working as desired)
                    if (sliceId < 2 && augmentIndex < 3 && fileIndex < 10)
                    {
                        using var inImg = greyTensor.ElementWise(px => px/255.0f).ToGreyscaleBitmaps();
                        using var inStream = File.Open(Path.Combine(ProcessedDataPath, baseName + ".slice" + sliceId + "." + augmentIndex + ".in" + ".png"), FileMode.Create);
                        inImg[0].Encode(inStream, SKEncodedImageFormat.Png, 100);
                        inStream.Flush();

                        var elementWiseColour = colourTensor.ElementWise(px => px/255.0f);
                        using var outImg = colourTensor.ElementWise(px => px/255.0f).ToColourBitmaps();
                        using var outStream = File.Open(Path.Combine(ProcessedDataPath, baseName + ".slice" + sliceId + "." + augmentIndex + ".out" + ".png"), FileMode.Create);
                        outImg[0].Encode(outStream, SKEncodedImageFormat.Png, 100);
                        outStream.Flush();

                        using var rawStream = File.Open(Path.Combine(ProcessedDataPath, baseName + ".slice" + sliceId + "." + augmentIndex + ".raw" + ".png"), FileMode.Create);
                        augment.Encode(rawStream, SKEncodedImageFormat.Png, 100);
                        rawStream.Flush();
                    }
                
                    augmentIndex++;
                }
            }

            // Perform validation augmentations
            {
                using var augments = augmentor.Augment(slice, augmentations: 5);

                foreach (var augment in augments)
                {
                    var greyTensor = augment.ToGreyscaleTensor();
                    foreach (var val in greyTensor.AsSpan())
                        vWriter.Write((byte)val);       // 0 - 255

                    var colourTensor = augment.ToColourTensor();
                    foreach (var val in colourTensor.AsSpan())
                        vWriter.Write((byte)val);       // 0 - 255
                }
            }

            sliceId++;
            }

            fileIndex++;
        }
    }

    public override void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation)
    {
        ListTrainingDataSource<float> trn = ParseTrainingFile("training.bin");
        ListTrainingDataSource<float> val = ParseTrainingFile("validation.bin");

        training    = trn;
        validation  = val;
    }
    private ListTrainingDataSource<float> ParseTrainingFile(string filename)
    {
        var ishape = new Shape(1, MinImgHeight, MinImgWidth);
        var oshape = new Shape(3, MinImgHeight, MinImgWidth);
        ListTrainingDataSource<float> trn = new ListTrainingDataSource<float>(ishape, oshape);

        using (var iReader = new BinaryReader(File.Open(Path.Combine(ProcessedDataPath, filename), FileMode.Open))) {
            while (iReader.BaseStream.Position < iReader.BaseStream.Length)
            {
                // Read input tensor
                Tensor<float> greyscale = Tensor<float>.Defaults(ishape);
                var greySpan = greyscale.AsSpan();
                for (var i = 0; i < greySpan.Length; i++)
                {
                    greySpan[i] = iReader.ReadByte() / 255.0f;
                }

                // Read output tensor
                Tensor<float> colour = Tensor<float>.Defaults(oshape);
                var colourSpan = colour.AsSpan();
                for (var i = 0; i < colourSpan.Length; i++)
                {
                    colourSpan[i] = iReader.ReadByte() / 255.0f;
                }

                // Add to dataset
                trn.Add((greyscale, colour));
            }
        }

        return trn;
    }
}
