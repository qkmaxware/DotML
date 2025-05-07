using DotML.Network;
using SkiaSharp;

namespace DotML.Cli.Logging;

public class ImagesOutputLogger : BaseOutputLogger {
    public ImagesOutputLogger(DirectoryInfo logDir) : base(logDir) { }

    private void EmitImage(string layerName, BatchedFeatureSet<double> args) {
        for (var batch = 0; batch < args.Batches; batch++) {
            var features = args[batch];
            var dir_path = Path.Combine(LogDirectory.FullName, $"Batch-{batch}");
            var dir = Directory.CreateDirectory(dir_path);

            var layer_dir_path = Path.Combine(dir_path, layerName);
            var layer_dir = Directory.CreateDirectory(layer_dir_path);

            var channels = features.Channels;
            for (var channel = 0; channel < channels; channel++) {
                var feature = features[channel];
                var width = feature.Columns;
                var height = feature.Rows;
                using var image = new SKBitmap(width, height, isOpaque: true);
                for (var y = 0; y < height; y++) {
                    for (var x = 0; x < width; x++) {
                        var offset = y * width + x;
                        var r = channels >= 1 ? (byte)Math.Clamp(feature[y, x] * 255, 0, 255) : (byte)0;
                        image.SetPixel(x, y, new SKColor(red: r, green: r, blue: r));
                    }
                }
                var file = Path.Combine(layer_dir_path, $"output.channel-{channel}.png");
                using (var stream = File.Open(file, FileMode.Create)) {
                    image.Encode(stream, SKEncodedImageFormat.Png, 100);
                }
            }
        }
    }

    public override void Visit(ConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(ConvolutionLayer)}", args.Output);
        return;
    }

    public override void Visit(DepthwiseConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(DepthwiseConvolutionLayer)}", args.Output);
        return;
    }

    public override void Visit(TransposeConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(TransposeConvolutionLayer)}", args.Output);
        return;
    }

    public override void Visit(PixelShuffle layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(PixelShuffle)}", args.Output);
        return;
    }

    public override void Visit(PoolingLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(PoolingLayer)}", args.Output);
        return;
    }

    public override void Visit(FlatteningLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(FlatteningLayer)}", args.Output);
        return;
    }

    public override void Visit(DropoutLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(DropoutLayer)}", args.Output);
        return;
    }

    public override void Visit(LayerNorm layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(LayerNorm)}", args.Output);
        return;
    }

    public override void Visit(BatchNorm layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(BatchNorm)}", args.Output);
        return;
    }

    public override void Visit(DenseLinearLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(DenseLinearLayer)}", args.Output);
        return;
    }

    public override void Visit(ActivationLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(ActivationLayer)}", args.Output);
        return;
    }

    public override void Visit(SoftmaxLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(SoftmaxLayer)}", args.Output);
        return;
    }

    public override void Visit(InputCapture capture, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return;
    }

    public override void Visit(AdditionSkipConnection capture, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(AdditionSkipConnection)}", args.Output);
        return;
    }

    public override void Visit(ConcatenationSkipConnection capture, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitImage($"Layer-{args.LayerIndex} {nameof(ConcatenationSkipConnection)}", args.Output);
        return;
    }
}