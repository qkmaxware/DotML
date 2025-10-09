namespace DotML.Network.Templates;

/// <summary>
/// Factory for creating the LeNet Convolutional Neural Network architecture
/// </summary>
public class UNetFactory : INetworkModuleFactory<UNetFactory.BuildSettings>
{
    public class BuildSettings
    {
        public int InputChannels { get; set; } = 3;      // e.g., RGB or RGB + timestep
        public int OutputChannels { get; set; } = 3;     // RGB
        public int BaseFeatureCount { get; set; } = 64;  // Number of channels in first layer
        public int Depth { get; set; } = 4;              // Number of encoder/decoder stages
        public int ImageWidth { get; set; } = 16;             // Optional, used for padding calc if needed must be power of 2 and larger than 2^depth for equal input and output image sizes
        public int ImageHeight { get; set; } = 16;
    }

    private static INetworkModule ConvBlock(int inChannels, int outChannels)
    {
        return new SequentialBlock([
            new Conv2D(outChannels, inChannels, 1, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1)),
            new Activation(ActivationFunctions.ReLU),
            new Conv2D(outChannels, outChannels, 1, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1)),
            new Activation(ActivationFunctions.ReLU)
        ]);
    }

    private INetworkModule BuildRecursiveUNet(
        int depth,
        int inChannels,
        int baseFeatures,
        int targetHeight,
        int targetWidth)
    {
        int currentFeatures = baseFeatures * (1 << (depth - 1));
        int nextFeatures = baseFeatures * (1 << depth);

        if (depth == 0)
        {
            // Bottleneck block: just a double conv
            return ConvBlock(inChannels, baseFeatures);
        }

        // Encoder block
        var encoder = ConvBlock(inChannels, currentFeatures);

        // Downsample
        var downsample = new MaxPool2D(size: 2, stride: 2, padding: 0);

        // Compute expected size after downsampling
        int downHeight = (targetHeight + 1) / 2;
        int downWidth = (targetWidth + 1) / 2;

        // Recursive descent
        var deeperUNet = BuildRecursiveUNet(
            depth - 1,
            inChannels: currentFeatures,
            baseFeatures: baseFeatures,
            targetHeight: downHeight,
            targetWidth: downWidth
        );

        // Upsample
        var upsample = new TransposeConv2D(
            outChannels: currentFeatures,
            inChannelsPerGroup: nextFeatures,
            groups: 1,
            kernel: (2, 2),
            stride: (2, 2),
            dilation: (1, 1),
            inputPadding: (0, 0, 0, 0),
            outputPadding: (0, 0, 0, 0)
        );

        // Decoder block (after concat, so channel count doubles)
        var decoder = ConvBlock(currentFeatures * 2, currentFeatures);

        // Compose main path
        var mainPath = new SequentialBlock([
            encoder,
            downsample,
            deeperUNet,
            upsample,
            decoder
        ]);

        // Add CenterCrop to align skip path with main path output
        var skipPath = new Center2D(rows: targetHeight, columns: targetWidth);

        return new ResidualConcat(axis: ^3, mainPath, skipPath);
    }

    public INetworkModule Make(BuildSettings settings)
    {
        var ishape = new TensorShape(Math.Max(1, settings.InputChannels), Math.Max(0, settings.ImageHeight),  Math.Max(0, settings.ImageWidth));

        // Build the full UNet recursively
        var unet = new ArchitectureBlock(
            name: "UNet",
            inputShape: ishape,
            rootModule: new SequentialBlock([
                BuildRecursiveUNet(
                    depth: Math.Max(1, settings.Depth),
                    inChannels: Math.Max(1, settings.InputChannels),
                    baseFeatures: Math.Max(1, settings.BaseFeatureCount),
                    targetHeight: Math.Max(0, settings.ImageHeight),
                    targetWidth: Math.Max(0, settings.ImageWidth)
                ),
                // Final 1x1 conv to map features to output RGB image
                new Conv2D(
                    outChannels: Math.Max(1, settings.OutputChannels),
                    inChannelsPerGroup: Math.Max(1, settings.BaseFeatureCount),
                    groups: 1,
                    kernel: (1, 1),
                    stride: (1, 1),
                    dilation: (1, 1),
                    padding: (0, 0, 0, 0)
                )
            ])
        );

        return unet;
    }
}

/*
using System;
using System.Drawing;
using System.IO;
using System.Linq;

public static class DatasetBuilder
{
    public static void GenerateTrainingDataset(
        string imageDirectory,
        int numNoiseLevels = 10,
        float noiseStdDev = 1.0f)
    {
        string[] imageFiles = Directory.GetFiles(imageDirectory, "*.png"); // or *.jpg

        Random rng = new Random();

        foreach (string imagePath in imageFiles)
        {
            using var bmp = new Bitmap(imagePath);
            var clean = bmp.ToTensor(normalize: true); // Shape: 3×H×W
            var target = clean;

            for (int step = 0; step < numNoiseLevels; step++)
            {
                float noiseLevel = 1.0f - (step / (float)(numNoiseLevels - 1));

                var noisy = AddNoise(clean, noiseLevel * noiseStdDev, rng); // still 3×H×W
                var noiseChannel = CreateNoiseLevelChannel(clean, noiseLevel); // 1×H×W

                var input = ConcatChannels(noisy, noiseChannel); // 4×H×W

                string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmssfff");

                string baseName = Path.GetFileNameWithoutExtension(imagePath);
                string inputPath = Path.Combine(imageDirectory, $"{baseName}_{timestamp}_step{step}.input.bin");
                string targetPath = Path.Combine(imageDirectory, $"{baseName}_{timestamp}_step{step}.target.bin");

                using (var inputStream = new BinaryWriter(File.OpenWrite(inputPath)))
                    input.SaveBinary(inputStream);

                using (var targetStream = new BinaryWriter(File.OpenWrite(targetPath)))
                    target.SaveBinary(targetStream);

                target = noisy; // So that the loop will work
            }
        }
    }

    private static Tensor<float> AddNoise(Tensor<float> image, float stdDev, Random rng)
    {
        var noisy = image.Clone();

        for (int c = 0; c < noisy.Shape.Length(0); c++)
        {
            for (int h = 0; h < noisy.Shape.Length(1); h++)
            {
                for (int w = 0; w < noisy.Shape.Length(2); w++)
                {
                    // Gaussian noise using Box-Muller transform
                    float u1 = 1.0f - (float)rng.NextDouble(); // avoid log(0)
                    float u2 = 1.0f - (float)rng.NextDouble();
                    float randStdNormal = (float)(Math.Sqrt(-2.0f * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));

                    float noise = randStdNormal * noiseStdDev;
                    float value = noisy[c, h, w] + noise;

                    // Clamp to 0–255
                    noisy[c, h, w] = Math.Clamp(value, 0f, 255f);
                }
            }
        }

        return noisy;
    }

    private static Tensor<float> CreateNoiseLevelChannel(Tensor<float> reference, float value)
    {
        int height = reference.Shape.Length(1);
        int width = reference.Shape.Length(2);

        var channel = new Tensor<float>(new TensorShape(1, height, width));

        for (int h = 0; h < height; h++)
            for (int w = 0; w < width; w++)
                channel[0, h, w] = value;

        return channel;
    }

    private static Tensor<float> ConcatChannels(Tensor<float> a, Tensor<float> b)
    {
        // Assumes both are shaped as C×H×W
        int channelsA = a.Shape.Length(0);
        int channelsB = b.Shape.Length(0);
        int height = a.Shape.Length(1);
        int width = a.Shape.Length(2);

        var result = new Tensor<float>(new TensorShape(channelsA + channelsB, height, width));

        for (int c = 0; c < channelsA; c++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                    result[c, h, w] = a[c, h, w];

        for (int c = 0; c < channelsB; c++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                    result[c + channelsA, h, w] = b[c, h, w];

        return result;
    }
}

DatasetBuilder.GenerateTrainingDataset(@"C:\training\images", numNoiseLevels: 10, noiseStdDev: 0.1f); // 0.02 = small, 0.1 = moderate, 0.2+ = heavy
*/