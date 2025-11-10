
using System.Text;

namespace DotML.Network.IO;

internal class NetbuildLayerDescriber : IBlockVisitor<string>
{
    private StringBuilder builder;

    public NetbuildLayerDescriber(StringBuilder builder)
    {
        this.builder = builder;
    }

    public string Visit(Activation activation, None arg)
    {
        var alpha = activation.ActivationFunction.GetType().GetProperty("Alpha")?.GetValue(activation.ActivationFunction);
        return $"{nameof(Activation)} fn={activation.ActivationFunction.GetType().Name} alpha={alpha ?? 0}";
    }

    public string Visit(SoftmaxOutput softmax, None arg)
    {
        return $"{nameof(SoftmaxOutput)} classesAxis={softmax.ClassAxis}";
    }

    public string Visit(Conv2D conv, None arg)
    {
        // public Conv2D(int outChannels, int inChannelsPerGroup, int groups, (int Width, int Height) kernel, (int X, int Y) stride, (int X, int Y) dilation, (int Left, int Top, int Right, int Bottom) padding)
        return $"{nameof(Conv2D)} outChannels={conv.Weights.Shape[0]} inChannelsPerGroup={conv.Weights.Shape[1]} groups={conv.Groups} kernel=\"{conv.Weights.Shape[3]},{conv.Weights.Shape[2]}\" stride=\"{conv.Stride.X},{conv.Stride.Y}\" dilation=\"{conv.Dilation.X},{conv.Dilation.Y}\" padding=\"{conv.Padding.Left},{conv.Padding.Top},{conv.Padding.Right},{conv.Padding.Bottom}\"";
    }

    public string Visit(TransposeConv2D conv, None arg)
    {
        //public TransposeConv2D(int outChannels, int inChannelsPerGroup, int groups, (int Width, int Height) kernel, (int X, int Y) stride, (int X, int Y) dilation, (int Left, int Top, int Right, int Bottom) inputPadding, (int Left, int Top, int Right, int Bottom) outputPadding)
        return $"{nameof(TransposeConv2D)} outChannels={conv.Biases.Shape[0]} inChannelsPerGroup={conv.Weights.Shape[0]} groups={conv.Groups} kernel=\"{conv.Weights.Shape[3]},{conv.Weights.Shape[2]}\" stride=\"{conv.Stride.X},{conv.Stride.Y}\" dilation=\"{conv.Dilation.X},{conv.Dilation.Y}\" inputPadding=\"{conv.InputPadding.Left},{conv.InputPadding.Top},{conv.InputPadding.Right},{conv.InputPadding.Bottom}\" outputPadding=\"{conv.OutputPadding.Left},{conv.OutputPadding.Top},{conv.OutputPadding.Right},{conv.OutputPadding.Bottom}\"";
    }

    public string Visit(DenseLinear dense, None arg)
    {
        return $"{nameof(DenseLinear)} input_size={dense.InputSize} neurons={dense.Neurons}";
    }

    public string Visit(Dropout dropout, None arg)
    {
        return $"{nameof(Dropout)} dropoutRate={dropout.DropoutRate}";
    }

    public string Visit(BatchNorm2D norm, None arg)
    {
        return $"{nameof(BatchNorm2D)} channels={norm.Biases.ElementCount}";
    }

    public string Visit(GroupNorm norm, None arg)
    {
        // public GroupNorm2(int num_groups, TensorShape normalizedShape)
        return $"{nameof(GroupNorm)} num_groups={norm.Groups} normalizedShape=\"{string.Join(',', norm.NormalizedShape.AsDimensionEnumerable())}\"";
    }

    public string Visit(LayerNorm norm, None arg)
    {
        // public LayerNorm2(TensorShape normalizedShape)
        return $"{nameof(LayerNorm)} normalizedShape=\"{string.Join(',', norm.NormalizedShape.AsDimensionEnumerable())}\"";
    }

    public string Visit(PixelShuffler shuffle, None arg)
    {
        return $"{nameof(PixelShuffler)} upscale={shuffle.UpscalingFactor}";
    }

    public string Visit(AvgPool2D pool, None arg)
    {
        return $"{nameof(AvgPool2D)} width={pool.FilterSize.Width} height={pool.FilterSize.Height} strideX={pool.Stride.X} strideY={pool.Stride.Y} paddingX={pool.PaddingX} paddingY={pool.PaddingY}";
    }

    public string Visit(MaxPool2D pool, None arg)
    {
        return $"{nameof(MaxPool2D)} width={pool.FilterSize.Width} height={pool.FilterSize.Height} strideX={pool.Stride.X} strideY={pool.Stride.Y} paddingX={pool.PaddingX} paddingY={pool.PaddingY}";
    }

    public string Visit(MinPool2D pool, None arg)
    {
        return $"{nameof(MinPool2D)} width={pool.FilterSize.Width} height={pool.FilterSize.Height} strideX={pool.Stride.X} strideY={pool.Stride.Y} paddingX={pool.PaddingX} paddingY={pool.PaddingY}";
    }

    public string Visit(GlobalAvgPool2D pool, None arg)
    {
        return $"{nameof(MinPool2D)}";
    }

    public string Visit(GlobalMaxPool2D pool, None arg)
    {
        return $"{nameof(GlobalMaxPool2D)}";
    }

    public string Visit(GlobalMinPool2D pool, None arg)
    {
        return $"{nameof(GlobalMinPool2D)}";
    }

    public string Visit(Reshape reshape, None arg)
    {
        throw new NotImplementedException();
    }

    public string Visit(Flatten flatten, None arg)
    {
        return $"{nameof(Flatten)} mode={flatten.Mode}";
    }

    public string Visit(Center2D center, None arg)
    {
        return $"{nameof(Center2D)} rows={center.TargetRows} columns={center.TargetColumns}";
    }

    public string Visit(ResidualBlock block, None arg)
    {
        throw new NotImplementedException();
    }

    public string Visit(SequentialBlock block, None arg)
    {
        for (var layerIndex = 0; layerIndex < block.LayerCount; layerIndex++)
        {
            var layer = block.GetLayer(layerIndex);
            if (layer is not IBlockVisitable visitable)
                throw new ArgumentException("All network layers must implement IBlockVisitable");

            var x = visitable.Accept(this);
            if (!string.IsNullOrEmpty(x))
            {
                builder.Append("ADD "); builder.AppendLine(x);
            }
        }

        return string.Empty;
    }
}