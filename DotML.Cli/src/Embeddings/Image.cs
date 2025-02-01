using SkiaSharp;

namespace DotML.Cli.Embeddings;

/// <summary>
/// Treat the input as an RGB image whose pixel values are representable by bytes between 0 and 255
/// </summary>
public class RgbImage : IEmbedder {
    public Vec<double> CreateEmbedding(FileInfo file) {
        using var bitmap = SKBitmap.FromImage(SKImage.FromEncodedData(file.FullName));
        var rows        = bitmap.Height;
        var cols        = bitmap.Width;
        var img_size    = rows * cols;
        var samples     = 3;
        var vector      = new Vec<double>(samples * img_size);

        for (var row = 0; row < rows; row++) {
            for (var col = 0; col < cols; col++) {
                var index_in_sample = row * cols + col;
                var colour = bitmap.GetPixel(col, row);

                vector[0 * img_size + index_in_sample] = (colour.Red / 255.0);
                vector[1 * img_size + index_in_sample] = (colour.Green / 255.0);
                vector[2 * img_size + index_in_sample] = (colour.Blue / 255.0);
            }
        }

        return vector;
    }

    public Vec<double> CreateEmbedding(string raw) {
        throw new NotSupportedException("Reading images from standard input is not supported");
    }
}


/// <summary>
/// Treat the input as a Mono image whose pixel values are representable by bytes between 0 and 255
/// </summary>
public class MonoImage : IEmbedder {
    public Vec<double> CreateEmbedding(FileInfo file) {
        using var bitmap = SKBitmap.FromImage(SKImage.FromEncodedData(file.FullName));
        var rows        = bitmap.Height;
        var cols        = bitmap.Width;
        var img_size    = rows * cols;
        var samples     = 3;
        var vector      = new Vec<double>(samples * img_size);

        for (var row = 0; row < rows; row++) {
            for (var col = 0; col < cols; col++) {
                var index_in_sample = row * cols + col;
                var colour = bitmap.GetPixel(col, row);

                var grey = (0.299 * colour.Red) + (0.587 * colour.Green) + (0.114 * colour.Blue);
                vector[index_in_sample] = Math.Clamp(grey, 0, 255) / 255.0;
            }
        }

        return vector;
    }

    public Vec<double> CreateEmbedding(string raw) {
        throw new NotSupportedException("Reading images from standard input is not supported");
    }
}