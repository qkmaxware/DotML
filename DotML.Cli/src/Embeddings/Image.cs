using DotML.Network;
using DotML.Network.Training;
using SkiaSharp;

namespace DotML.Cli.Embeddings;

/// <summary>
/// Treat the input as an image whose pixel values are representable by bytes between 0 and 255
/// </summary>
public class Image : ImageEmbedding {
    public override FeatureSet<double> CreateEmbedding(FeedforwardNetwork network, SKBitmap bitmap) {
        var rows        = bitmap.Height;
        var cols        = bitmap.Width;
        var samples     = network.InputShape.Channels;

        const int R = 0;
        const int G = 1;
        const int B = 2;
        var features = new FeatureSet<double>(new Shape3D(samples, rows, cols));

        for (var row = 0; row < rows; row++) {
            for (var col = 0; col < cols; col++) {
                var index_in_sample = row * cols + col;
                var colour = bitmap.GetPixel(col, row);

                if (samples >= 1)
                    features[R, row, col] = (colour.Red / 255.0);
                if (samples >= 2)
                    features[G, row, col] = (colour.Green / 255.0);
                if (samples >= 3)
                    features[B, row, col] = (colour.Blue / 255.0);
            }
        }

        return features;
    }
}

/// <summary>
/// Treat the input as an RGB image whose pixel values are representable by bytes between 0 and 255
/// </summary>
public class RgbImage : ImageEmbedding {
    public override FeatureSet<double> CreateEmbedding(FeedforwardNetwork network, SKBitmap bitmap) {
        var rows        = bitmap.Height;
        var cols        = bitmap.Width;
        var samples     = 3;

        const int R = 0;
        const int G = 1;
        const int B = 2;
        var features = new FeatureSet<double>(new Shape3D(samples, rows, cols));

        for (var row = 0; row < rows; row++) {
            for (var col = 0; col < cols; col++) {
                var index_in_sample = row * cols + col;
                var colour = bitmap.GetPixel(col, row);

                features[R, row, col] = (colour.Red / 255.0);
                features[G, row, col] = (colour.Green / 255.0);
                features[B, row, col] = (colour.Blue / 255.0);
            }
        }

        return features;
    }
}


/// <summary>
/// Treat the input as a Mono image whose pixel values are representable by bytes between 0 and 255
/// </summary>
public class MonoImage : ImageEmbedding {
    public override FeatureSet<double> CreateEmbedding(FeedforwardNetwork network, SKBitmap bitmap) {
        var rows        = bitmap.Height;
        var cols        = bitmap.Width;
        var samples     = 1;

        const int GREY = 0;
        var features = new FeatureSet<double>(new Shape3D(samples, rows, cols));

        for (var row = 0; row < rows; row++) {
            for (var col = 0; col < cols; col++) {
                var index_in_sample = row * cols + col;
                var colour = bitmap.GetPixel(col, row);

                var grey = (0.299 * colour.Red) + (0.587 * colour.Green) + (0.114 * colour.Blue);
                features[GREY, row, col] = Math.Clamp(grey, 0, 255) / 255.0;
            }
        }

        return features;
    }
}