
using DotML.Network;
using SkiaSharp;

namespace DotML.Cli.Embeddings;

/// <summary>
/// An embedding that must be provided by a file and not accessible via standard input
/// </summary>
public abstract class FileOnlyEmbedding : IEmbedder {
    public abstract Tensor<float> CreateEmbedding(INetworkModule @for, FileInfo file);

    public Tensor<float> CreateEmbedding(INetworkModule @for, string raw) {
        throw new NotSupportedException($"{GetType()} embedding doesn't support data from stdin.");
    }
}

/// <summary>
/// Base class for all embeddings that operate on images
/// </summary>
public abstract class ImageEmbedding : FileOnlyEmbedding {

    public override Tensor<float> CreateEmbedding(INetworkModule @for, FileInfo file) {
        using var original = SKBitmap.FromImage(SKImage.FromEncodedData(file.FullName));
        using var processed = PreprocessImage(@for, original);
        var features = CreateEmbedding(@for, processed);
        var postprocessed = Postprocess(features);
        return postprocessed;
    }

    public virtual SKBitmap PreprocessImage(INetworkModule @for, SKBitmap original) {
        // Crop image to match aspect ratio
        var aspect = (double)original.Width / (double)original.Height;
        var ishape = @for is ArchitectureBlock arch && arch.RequiredInputShape.HasValue 
            ? new Shape3D(arch.RequiredInputShape.Value.LengthOrDefault(^3), arch.RequiredInputShape.Value.LengthOrDefault(^2), arch.RequiredInputShape.Value.LengthOrDefault(^1)) 
            : new Shape3D(3, original.Height, original.Width);

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

    public abstract Tensor<float> CreateEmbedding(INetworkModule @for, SKBitmap bitmap);

    public virtual Tensor<float> Postprocess(Tensor<float> features) {
        return features;
    }

}