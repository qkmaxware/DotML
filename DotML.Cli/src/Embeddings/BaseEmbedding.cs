
using DotML.Network;
using SkiaSharp;

namespace DotML.Cli.Embeddings;

/// <summary>
/// An embedding that must be provided by a file and not accessible via standard input
/// </summary>
public abstract class FileOnlyEmbedding : IEmbedder {
    public abstract BatchedFeatureSet<double> CreateEmbedding(FeedforwardNetwork @for, IEnumerable<FileInfo> files);

    public BatchedFeatureSet<double> CreateEmbedding(FeedforwardNetwork @for, string raw) {
        throw new NotSupportedException($"{GetType()} embedding doesn't support data from stdin.");
    }
}

/// <summary>
/// Base class for all embeddings that operate on images
/// </summary>
public abstract class ImageEmbedding : FileOnlyEmbedding {

    public override BatchedFeatureSet<double> CreateEmbedding(FeedforwardNetwork @for, IEnumerable<FileInfo> files) {
        var batches = files.Select(file => CreateEmbedding(@for, file)).ToArray();
        return new BatchedFeatureSet<double>(batches);
    }

    public FeatureSet<double> CreateEmbedding(FeedforwardNetwork @for, FileInfo file) {
        using var original = SKBitmap.FromImage(SKImage.FromEncodedData(file.FullName));
        using var processed = PreprocessImage(@for, original);
        var features = CreateEmbedding(@for, processed);
        var postprocessed = Postprocess(features);
        return postprocessed;
    }

    public virtual SKBitmap PreprocessImage(FeedforwardNetwork @for, SKBitmap original) {
        // Crop image to match aspect ratio
        var aspect = (double)original.Width / (double)original.Height;
        var ishape = @for.InputShape;

        var desired_aspect = ishape.Columns / ishape.Rows;
        var cropWidth = original.Width;
        var cropHeight = (int)(original.Width / desired_aspect);

        if (cropHeight > original.Height) {
            cropHeight = original.Height;
            cropWidth = (int)(original.Height * desired_aspect);
        }

        int x_offset = (original.Width - cropWidth) / 2;
        int y_offset = (original.Height - cropHeight) / 2;

        var cropArea = new SKRectI(
            x_offset,               y_offset,
            x_offset + cropWidth,   y_offset + cropHeight
        );

        using var cropped = new SKBitmap(cropWidth, cropHeight);
        using (var canvas = new SKCanvas(cropped)) {
            canvas.DrawBitmap(original, new SKRect(-x_offset, -y_offset, original.Width - x_offset, original.Height - y_offset));
        }

        // Scale image to match input size
        #pragma warning disable CS0618
        var resized = cropped.Resize(
            new SKSizeI(ishape.Columns, ishape.Rows), 
            SKFilterQuality.High
        );
        #pragma warning restore CS0618
        return resized;
    }

    public abstract FeatureSet<double> CreateEmbedding(FeedforwardNetwork @for, SKBitmap bitmap);

    public virtual FeatureSet<double> Postprocess(FeatureSet<double> features) {
        return features;
    }

}