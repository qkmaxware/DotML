using DotML.Network;
using DotML.Network.Training;
using SkiaSharp;

namespace DotML.Cli.Embeddings;

/// <summary>
/// Treat the input as an image whose pixel values are representable by bytes between 0 and 255
/// </summary>
public class Image : ImageEmbedding {
    public override Tensor<float> CreateEmbedding(INetworkModule network, SKBitmap bitmap) {
        var rows        = bitmap.Height;
        var cols        = bitmap.Width;
        var samples     = bitmap.ColorType == SKColorType.Gray8 ? 1 : 3;

        const int R = 0;
        const int G = 1;
        const int B = 2;
        var features = Tensor<float>.Defaults(new TensorShape(samples, rows, cols));

        for (var row = 0; row < rows; row++) {
            for (var col = 0; col < cols; col++) {
                var index_in_sample = row * cols + col;
                var colour = bitmap.GetPixel(col, row);

                if (samples >= 1)
                    features[R, row, col] = (colour.Red / 255.0f);
                if (samples >= 2)
                    features[G, row, col] = (colour.Green / 255.0f);
                if (samples >= 3)
                    features[B, row, col] = (colour.Blue / 255.0f);
            }
        }

        return features;
    }
}

/// <summary>
/// Treat the input as an RGB image whose pixel values are representable by bytes between 0 and 255
/// </summary>
public class RgbImage : ImageEmbedding {
    public override Tensor<float> CreateEmbedding(INetworkModule network, SKBitmap bitmap) {
        var rows        = bitmap.Height;
        var cols        = bitmap.Width;
        var samples     = 3;

        const int R = 0;
        const int G = 1;
        const int B = 2;
        var features = Tensor<float>.Defaults(new TensorShape(samples, rows, cols));

        for (var row = 0; row < rows; row++) {
            for (var col = 0; col < cols; col++) {
                var index_in_sample = row * cols + col;
                var colour = bitmap.GetPixel(col, row);

                features[R, row, col] = (colour.Red / 255.0f);
                features[G, row, col] = (colour.Green / 255.0f);
                features[B, row, col] = (colour.Blue / 255.0f);
            }
        }

        return features;
    }
}


/// <summary>
/// Treat the input as a Mono image whose pixel values are representable by bytes between 0 and 255
/// </summary>
public class MonoImage : ImageEmbedding {
    public override Tensor<float> CreateEmbedding(INetworkModule network, SKBitmap bitmap) {
        var rows        = bitmap.Height;
        var cols        = bitmap.Width;
        var samples     = 1;

        const int GREY = 0;
        var features = Tensor<float>.Defaults(new Shape3D(samples, rows, cols));

        for (var row = 0; row < rows; row++) {
            for (var col = 0; col < cols; col++) {
                var index_in_sample = row * cols + col;
                var colour = bitmap.GetPixel(col, row);

                var grey = (0.299f * colour.Red) + (0.587f * colour.Green) + (0.114f * colour.Blue);
                features[GREY, row, col] = Math.Clamp(grey, 0, 255) / 255.0f;
            }
        }

        return features;
    }
}