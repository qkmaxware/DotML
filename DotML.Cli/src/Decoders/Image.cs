using DotML.Network.Training;
using SkiaSharp;

namespace DotML.Cli.Decodings;

/// <summary>
/// Decode an output into an image file
/// </summary>
public class Image : IDecoder {

    public class Result : IDecodedResult {
        
        int channels;
        SKBitmap[] bitmaps;

        public Result(int channels, SKBitmap[] bitmaps) {
            this.channels = channels;
            this.bitmaps = bitmaps;
        }

        public void ConsoleOutput() {
            int i = 1;
            foreach (var bitmap in bitmaps) {
                Console.WriteLine($"Image {i++}: A {channels}-channel {bitmap.Width}x{bitmap.Height} image.");
            }
        }

        public void FileOutput(FileInfo file) {
            if (file.Extension != ".png") {
                file = new FileInfo(file.FullName + ".png");
            }

            if (bitmaps.Length == 1) {
                var bitmap = bitmaps[0];
                using (var stream = File.Open(file.FullName, FileMode.Create)) {
                    bitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
                }
            } else {
                var path = file.FullName;
                for (var i = 0; i < bitmaps.Length; i++) {
                    var save_to = Path.ChangeExtension(path, $".{i}.png"); // Number the images if there are more than 1
                    var bitmap = bitmaps[i];
                    using (var stream = File.Open(save_to, FileMode.Create)) {
                        bitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
                    }
                }
            }
        }

        public void Dispose() {
            foreach (var bitmap in bitmaps)
                bitmap.Dispose();
        }
    }

    public IDecodedResult Decode(BatchedFeatureSet<double> output) {
        var images = new SKBitmap[output.Batches];
        var channels = Math.Max(output.Channels, 3);
        for (var batch = 0; batch < images.Length; batch++) {
            var features = output[batch];
            var output_shape = features.Shape;
            var width = output_shape.Columns;
            var height = output_shape.Rows;
            var size = width * height;
            var bitmap = new SKBitmap(width,height, isOpaque: true);  
            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    var offset = y * width + x;
                    var r = channels >= 1 ? (byte)Math.Clamp(features[0, y, x] * 255, 0, 255) : (byte)0;
                    var g = channels >= 2 ? (byte)Math.Clamp(features[1, y, x] * 255, 0, 255) : (byte)0;
                    var b = channels >= 3 ? (byte)Math.Clamp(features[2, y, x] * 255, 0, 255) : (byte)0;
                    if (channels == 1) {
                        g = r; b = r; // For mono-images use the same colour for all 3 components
                    }
                    bitmap.SetPixel(x, y, new SKColor(red: r, green: g, blue: b));
                }
            }
            images[batch] = bitmap;
        }
        return new Result(channels, images);
    }
}
