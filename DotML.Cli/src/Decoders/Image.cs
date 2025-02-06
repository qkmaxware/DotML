using SkiaSharp;

namespace DotML.Cli.Decodings;


public class Image : IDecoder {

    public class Result : IDecodedResult {
        
        int channels;
        SKBitmap bitmap;

        public Result(int channels, SKBitmap bitmap) {
            this.channels = channels;
            this.bitmap = bitmap;
        }

        public void ConsoleOutput() {
            Console.WriteLine($"A {channels}-channel {bitmap.Width}x{bitmap.Height} image.");
        }

        public void FileOutput(FileInfo file) {
            if (file.Extension != ".png") {
                file = new FileInfo(file.FullName + ".png");
            }

            bitmap.Encode(file.OpenWrite(), SKEncodedImageFormat.Png, 100);
        }

        public void Dispose() {
            bitmap.Dispose();
        }
    }

    public IDecodedResult Decode(Shape3D output_shape, Vec<double> output) {
        var width = output_shape.Columns;
        var height = output_shape.Rows;
        var size = width * height;
        var bitmap = new SKBitmap(width,height, isOpaque: true);
        var channels = Math.Max(output_shape.Channels, 3);
        for (var y = 0; y < height; y++) {
            for (var x = 0; x < width; x++) {
                var offset = y * width + x;
                var r = channels >= 1 ? (byte)output[0 * size + offset] : (byte)0;
                var g = channels >= 2 ? (byte)output[1 * size + offset] : (byte)0;
                var b = channels >= 3 ? (byte)output[2 * size + offset] : (byte)0;
                if (channels == 1) {
                    g = r; b = r; // For mono-images use the same colour for all 3 components
                }
                bitmap.SetPixel(x, y, new SKColor(red: r, green: g, blue: b));
            }
        }
        return new Result(channels, bitmap);
    }
}
