using DotML.Network;
using SkiaSharp;

namespace DotML.Cli.Logging;

public class ImagesOutputLogger : BaseOutputLogger {
    public ImagesOutputLogger(DirectoryInfo logDir) : base(logDir) { }

    public override void Log(string identifier, Tensor<float> output)
    {
        output = output.ReshapeShared(output.Shape.NormalizeRank(4)); // NCHW
        var batches = output.Shape[0];
        var channels = output.Shape[0];
        var height = output.Shape[0];
        var width = output.Shape[0];

        var dir_path = Path.Combine(LogDirectory.FullName, identifier);
        var dir = Directory.CreateDirectory(dir_path);

        for (var batch = 0; batch < batches; batch++)
        {
            for (var channel = 0; channel < channels; channel++)
            {
                using var image = new SKBitmap(width, height, isOpaque: true);
                for (var y = 0; y < height; y++) {
                    for (var x = 0; x < width; x++) {
                        var offset = y * width + x;
                        var r = channels >= 1 ? (byte)Math.Clamp(output[batch, channel, y, x] * 255, 0, 255) : (byte)0;
                        image.SetPixel(x, y, new SKColor(red: r, green: r, blue: r));
                    }
                }
                var file = Path.Combine(dir.FullName, $"output.batch-{batch}.channel-{channel}.png");
                using (var stream = File.Open(file, FileMode.Create)) {
                    image.Encode(stream, SKEncodedImageFormat.Png, 100);
                }
            }
        }
    }

}