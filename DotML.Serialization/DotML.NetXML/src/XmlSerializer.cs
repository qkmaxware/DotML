using System.Linq.Expressions;
using System.Xml;
using DotML.Network;

namespace DotML.Serialization.Xml;

public class NetworkXmlSerializer
: IBlockVisitor<XmlDocument, XmlElement>        // For serialization)
{
    public XmlDocument Serialize(IBlockVisitable visitable)
    {
        var doc = new XmlDocument();
        var root = visitable.Accept(this, doc);
        doc.AppendChild(root);
        return doc;
    }

    public INetworkModule Deserialize(XmlDocument document)
    {
        var root = document.DocumentElement;
        if (root is null)
            throw new NullReferenceException(nameof(XmlDocument.DocumentElement));
        return Deserialize(root);
    }

    private INetworkModule Deserialize(XmlElement? element)
    {
        if (element is null)
            throw new NullReferenceException(nameof(element));

        return element.Name switch
        {
            nameof(Activation) => DecodeActivation(element),
            nameof(SoftmaxOutput) => DecodeSoftmaxOutput(element),
            nameof(Conv2D) => DecodeConv2D(element),
            nameof(TransposeConv2D) => DecodeTransposeConv2D(element),
            nameof(DenseLinear) => DecodeDenseLinear(element),
            nameof(Dropout) => DecodeDropout(element),
            nameof(BatchNorm2D) => DecodeBatchNorm2D(element),
            nameof(GroupNorm) => DecodeGroupNorm(element),
            nameof(LayerNorm) => DecodeLayerNorm(element),
            nameof(PixelShuffler) => DecodePixelShuffler(element),
            nameof(AvgPool2D) => DecodeAvgPool2D(element),
            nameof(MaxPool2D) => DecodeMaxPool2D(element),
            nameof(MinPool2D) => DecodeMinPool2D(element),
            nameof(GlobalAvgPool2D) => DecodeGlobalAvgPool2D(element),
            nameof(GlobalMaxPool2D) => DecodeGlobalMaxPool2D(element),
            nameof(GlobalMinPool2D) => DecodeGlobalMinPool2D(element),
            nameof(Flatten) => DecodeFlatten(element),
            nameof(Center2D) => DecodeCenter2D(element),

            nameof(ResidualBlock) => DecodeResidualBlock(element),
            nameof(SequentialBlock) => DecodeSequentialBlock(element),
            _ => throw new NotImplementedException(element.Name)
        };
    }

    public XmlElement Visit(Activation activation, XmlDocument doc)
    {
        var elem = doc.CreateElement(nameof(Activation));
        var alpha = activation.ActivationFunction.GetType().GetProperty("Alpha")?.GetValue(activation.ActivationFunction);

        elem.SetAttribute("fn", activation.ActivationFunction.GetType().Name);
        if (alpha is not null)
            elem.SetAttribute("alpha", alpha.ToString());

        return elem;
    }
    public INetworkModule DecodeActivation(XmlElement el)
    {
        var fnName = el.GetAttribute("fn");
        var alpha = 0.0f;
        float.TryParse(el.GetAttribute("alpha"), out alpha);
        var fn = ActivationFunctionMapper.DecodeStatic(fnName, alpha);
        return new Activation(fn);
    }

    public XmlElement Visit(SoftmaxOutput softmax, XmlDocument doc)
    {
        var elem = doc.CreateElement(nameof(SoftmaxOutput));
        elem.SetAttribute("classes-axis", softmax.ClassAxis.ToString());
        return elem;
    }
    public INetworkModule DecodeSoftmaxOutput(XmlElement el)
    {
        var axis = el.GetAttribute("classes-axis");
        if (axis is null)
            throw new NullReferenceException("classes-axis");

        bool fromEnd = axis.StartsWith('^');
        var span = axis.AsSpan();
        if (fromEnd)
            span = span.Slice(1);
        var value = int.Parse(span, System.Globalization.NumberStyles.Integer);
        var index = new Index(value, fromEnd: fromEnd);
        return new SoftmaxOutput(index);
    }

    public XmlElement Visit(Conv2D conv, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(Conv2D));
        el.SetAttribute("out-channels", conv.Weights.Shape[0].ToString());
        el.SetAttribute("in-channels-per-group", conv.Weights.Shape[1].ToString());
        el.SetAttribute("groups", conv.Groups.ToString());
        el.SetAttribute("kernel-width", conv.Weights.Shape[3].ToString());
        el.SetAttribute("kernel-height", conv.Weights.Shape[2].ToString());
        el.SetAttribute("stride-x", conv.Stride.X.ToString());
        el.SetAttribute("stride-y", conv.Stride.Y.ToString());
        el.SetAttribute("dilation-x", conv.Dilation.X.ToString());
        el.SetAttribute("dilation-y", conv.Dilation.Y.ToString());
        el.SetAttribute("padding-left", conv.Padding.Left.ToString());
        el.SetAttribute("padding-top", conv.Padding.Top.ToString());
        el.SetAttribute("padding-right", conv.Padding.Right.ToString());
        el.SetAttribute("padding-bottom", conv.Padding.Bottom.ToString());
        return el;
    }
    public INetworkModule DecodeConv2D(XmlElement el)
    {
        int oc = int.Parse(el.GetAttribute("out-channels"));
        int ic = int.Parse(el.GetAttribute("in-channels-per-group"));
        int g = int.Parse(el.GetAttribute("groups"));
        int kw = int.Parse(el.GetAttribute("kernel-width"));
        int kh = int.Parse(el.GetAttribute("kernel-height"));
        int sx = int.Parse(el.GetAttribute("stride-x"));
        int sy = int.Parse(el.GetAttribute("stride-y"));
        int dx = int.Parse(el.GetAttribute("dilation-x"));
        int dy = int.Parse(el.GetAttribute("dilation-y"));
        int pl = int.Parse(el.GetAttribute("padding-left"));
        int pt = int.Parse(el.GetAttribute("padding-top"));
        int pr = int.Parse(el.GetAttribute("padding-right"));
        int pb = int.Parse(el.GetAttribute("padding-bottom"));

        return new Conv2D(oc, ic, g, new Size2D(kw, kh), new Stride2D(sx, sy), new Dilation2D(dx, dy), new Padding2D(pl, pt, pr, pb));
    }

    public XmlElement Visit(TransposeConv2D conv, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(TransposeConv2D));
        el.SetAttribute("out-channels", conv.Biases.Shape[0].ToString());
        el.SetAttribute("in-channels-per-group", conv.Weights.Shape[0].ToString());
        el.SetAttribute("groups", conv.Groups.ToString());
        el.SetAttribute("kernel-width", conv.Weights.Shape[3].ToString());
        el.SetAttribute("kernel-height", conv.Weights.Shape[2].ToString());
        el.SetAttribute("stride-x", conv.Stride.X.ToString());
        el.SetAttribute("stride-y", conv.Stride.Y.ToString());
        el.SetAttribute("dilation-x", conv.Dilation.X.ToString());
        el.SetAttribute("dilation-y", conv.Dilation.Y.ToString());
        el.SetAttribute("input-padding-left", conv.InputPadding.Left.ToString());
        el.SetAttribute("input-padding-top", conv.InputPadding.Top.ToString());
        el.SetAttribute("input-padding-right", conv.InputPadding.Right.ToString());
        el.SetAttribute("input-padding-bottom", conv.InputPadding.Bottom.ToString());
        el.SetAttribute("output-padding-left", conv.OutputPadding.Left.ToString());
        el.SetAttribute("output-padding-top", conv.OutputPadding.Top.ToString());
        el.SetAttribute("output-padding-right", conv.OutputPadding.Right.ToString());
        el.SetAttribute("output-padding-bottom", conv.OutputPadding.Bottom.ToString());
        return el;
    }
    public INetworkModule DecodeTransposeConv2D(XmlElement el)
    {
        int oc = int.Parse(el.GetAttribute("out-channels"));
        int ic = int.Parse(el.GetAttribute("in-channels-per-group"));
        int g = int.Parse(el.GetAttribute("groups"));
        int kw = int.Parse(el.GetAttribute("kernel-width"));
        int kh = int.Parse(el.GetAttribute("kernel-height"));
        int sx = int.Parse(el.GetAttribute("stride-x"));
        int sy = int.Parse(el.GetAttribute("stride-y"));
        int dx = int.Parse(el.GetAttribute("dilation-x"));
        int dy = int.Parse(el.GetAttribute("dilation-y"));
        int ipl = int.Parse(el.GetAttribute("input-padding-left"));
        int ipt = int.Parse(el.GetAttribute("input-padding-top"));
        int ipr = int.Parse(el.GetAttribute("input-padding-right"));
        int ipb = int.Parse(el.GetAttribute("input-padding-bottom"));
        int opl = int.Parse(el.GetAttribute("output-padding-left"));
        int opt = int.Parse(el.GetAttribute("output-padding-top"));
        int opr = int.Parse(el.GetAttribute("output-padding-right"));
        int opb = int.Parse(el.GetAttribute("output-padding-bottom"));

        return new TransposeConv2D(oc, ic, g, new Size2D(kw, kh), new Stride2D(sx, sy), new Dilation2D(dx, dy),
            new Padding2D(ipl, ipt, ipr, ipb),
            new Padding2D(opl, opt, opr, opb)
        );
    }

    public XmlElement Visit(DenseLinear dense, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(DenseLinear));
        el.SetAttribute("input", dense.InputSize.ToString());
        el.SetAttribute("output", dense.Neurons.ToString());
        return el;
    }
    public INetworkModule DecodeDenseLinear(XmlElement el)
    {
        var i = int.Parse(el.GetAttribute("input"));
        var o = int.Parse(el.GetAttribute("output"));
        return new DenseLinear(i, o);
    }

    public XmlElement Visit(Dropout dropout, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(Dropout));
        el.SetAttribute("rate", dropout.DropoutRate.ToString());
        return el;
    }
    public INetworkModule DecodeDropout(XmlElement el)
    {
        float d = float.Parse(el.GetAttribute("rate"));
        return new Dropout(d);
    }

    public XmlElement Visit(BatchNorm2D norm, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(BatchNorm2D));
        el.SetAttribute("channels", norm.Biases.ElementCount.ToString());
        return el;
    }
    public INetworkModule DecodeBatchNorm2D(XmlElement el)
    {
        int c = int.Parse(el.GetAttribute("channels"));
        return new BatchNorm2D(c);
    }

    public XmlElement Visit(GroupNorm norm, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(GroupNorm));
        el.SetAttribute("groups", norm.Groups.ToString());
        el.SetAttribute("normalized-shape", string.Join(';', norm.NormalizedShape.AsDimensionEnumerable()));
        return el;
    }
    public INetworkModule DecodeGroupNorm(XmlElement el)
    {
        int g = int.Parse(el.GetAttribute("groups"));
        int[] s = el.GetAttribute("normalized-shape").Split(';', StringSplitOptions.TrimEntries).Select(x => int.Parse(x)).ToArray();
        return new GroupNorm(g, new TensorShape(s));
    }

    public XmlElement Visit(LayerNorm norm, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(GroupNorm));
        el.SetAttribute("normalized-shape", string.Join(';', norm.NormalizedShape.AsDimensionEnumerable()));
        return el;
    }
    public INetworkModule DecodeLayerNorm(XmlElement el)
    {
        int[] s = el.GetAttribute("normalized-shape").Split(';', StringSplitOptions.TrimEntries).Select(x => int.Parse(x)).ToArray();
        return new LayerNorm(new TensorShape(s));
    }

    public XmlElement Visit(PixelShuffler shuffle, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(PixelShuffler));
        el.SetAttribute("upscale", shuffle.UpscalingFactor.ToString());
        return el;
    }
    public INetworkModule DecodePixelShuffler(XmlElement el)
    {
        int r = int.Parse(el.GetAttribute("upscale"));
        return new PixelShuffler(r);
    }

    public XmlElement Visit(AvgPool2D pool, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(AvgPool2D));
        el.SetAttribute("kernel-width", pool.FilterSize.Width.ToString());
        el.SetAttribute("kernel-height", pool.FilterSize.Height.ToString());
        el.SetAttribute("stride-x", pool.Stride.X.ToString());
        el.SetAttribute("stride-y", pool.Stride.Y.ToString());
        el.SetAttribute("padding-x", pool.PaddingX.ToString());
        el.SetAttribute("padding-y", pool.PaddingY.ToString());
        return el;
    }
    public INetworkModule DecodeAvgPool2D(XmlElement el)
    {
        int w = int.Parse(el.GetAttribute("kernel-width"));
        int h = int.Parse(el.GetAttribute("kernel-height"));
        int sx = int.Parse(el.GetAttribute("stride-x"));
        int sy = int.Parse(el.GetAttribute("stride-y"));
        int px = int.Parse(el.GetAttribute("padding-x"));
        int py = int.Parse(el.GetAttribute("padding-y"));
        return new AvgPool2D(w, h, sx, sy, px, py);
    }

    public XmlElement Visit(MaxPool2D pool, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(MaxPool2D));
        el.SetAttribute("kernel-width", pool.FilterSize.Width.ToString());
        el.SetAttribute("kernel-height", pool.FilterSize.Height.ToString());
        el.SetAttribute("stride-x", pool.Stride.X.ToString());
        el.SetAttribute("stride-y", pool.Stride.Y.ToString());
        el.SetAttribute("padding-x", pool.PaddingX.ToString());
        el.SetAttribute("padding-y", pool.PaddingY.ToString());
        return el;
    }
    public INetworkModule DecodeMaxPool2D(XmlElement el)
    {
        int w = int.Parse(el.GetAttribute("kernel-width"));
        int h = int.Parse(el.GetAttribute("kernel-height"));
        int sx = int.Parse(el.GetAttribute("stride-x"));
        int sy = int.Parse(el.GetAttribute("stride-y"));
        int px = int.Parse(el.GetAttribute("padding-x"));
        int py = int.Parse(el.GetAttribute("padding-y"));
        return new MaxPool2D(w, h, sx, sy, px, py);
    }

    public XmlElement Visit(MinPool2D pool, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(MinPool2D));
        el.SetAttribute("kernel-width", pool.FilterSize.Width.ToString());
        el.SetAttribute("kernel-height", pool.FilterSize.Height.ToString());
        el.SetAttribute("stride-x", pool.Stride.X.ToString());
        el.SetAttribute("stride-y", pool.Stride.Y.ToString());
        el.SetAttribute("padding-x", pool.PaddingX.ToString());
        el.SetAttribute("padding-y", pool.PaddingY.ToString());
        return el;
    }
    public INetworkModule DecodeMinPool2D(XmlElement el)
    {
        int w = int.Parse(el.GetAttribute("kernel-width"));
        int h = int.Parse(el.GetAttribute("kernel-height"));
        int sx = int.Parse(el.GetAttribute("stride-x"));
        int sy = int.Parse(el.GetAttribute("stride-y"));
        int px = int.Parse(el.GetAttribute("padding-x"));
        int py = int.Parse(el.GetAttribute("padding-y"));
        return new MinPool2D(w, h, sx, sy, px, py);
    }

    public XmlElement Visit(GlobalAvgPool2D pool, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(GlobalAvgPool2D));
        return el;
    }
    public INetworkModule DecodeGlobalAvgPool2D(XmlElement el)
    {
        return new GlobalAvgPool2D();
    }

    public XmlElement Visit(GlobalMaxPool2D pool, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(GlobalMaxPool2D));
        return el;
    }
    public INetworkModule DecodeGlobalMaxPool2D(XmlElement el)
    {
        return new GlobalMaxPool2D();
    }

    public XmlElement Visit(GlobalMinPool2D pool, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(GlobalMinPool2D));
        return el;
    }
    public INetworkModule DecodeGlobalMinPool2D(XmlElement el)
    {
        return new GlobalMinPool2D();
    }

    public XmlElement Visit(Reshape reshape, XmlDocument doc)
    {
        throw new NotImplementedException();
    }

    public XmlElement Visit(Flatten flatten, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(Flatten));
        el.SetAttribute(nameof(Flatten.Mode), flatten.Mode.ToString());
        return el;
    }
    public INetworkModule DecodeFlatten(XmlElement el)
    {
        var mode = Enum.Parse<Flatten.FlatteningMode>(el.GetAttribute(nameof(Flatten.Mode)));
        return new Flatten(mode);
    }

    public XmlElement Visit(Center2D center, XmlDocument doc)
    {
        var el = doc.CreateElement(nameof(Center2D));
        el.SetAttribute("rows", center.TargetRows.ToString());
        el.SetAttribute("columns", center.TargetColumns.ToString());
        return el;
    }
    public INetworkModule DecodeCenter2D(XmlElement el)
    {
        var r = int.Parse(el.GetAttribute("rows"));
        var c = int.Parse(el.GetAttribute("columns"));
        return new Center2D(r, c);
    }

    public XmlElement Visit(ResidualBlock block, XmlDocument doc)
    {
        var elem = doc.CreateElement(nameof(ResidualBlock));
        elem.SetAttribute("Type", block.GetType().Name);

        var skipElem = doc.CreateElement(nameof(ResidualBlock.ResidualPath));
        elem.AppendChild(skipElem);

        var mainElem = doc.CreateElement(nameof(ResidualBlock.MainPath));
        elem.AppendChild(mainElem);

        if (block.MainPath is IBlockVisitable main)
        {
            var mainSeq = main.Accept(this, doc);
            mainElem.AppendChild(mainSeq);
        }

        if (block.ResidualPath is IBlockVisitable skip)
        {
            var skipSeq = skip.Accept(this, doc);
            skipElem.AppendChild(skipSeq);
        }

        return elem;
    }
    public INetworkModule DecodeResidualBlock(XmlElement el)
    {
        var mainPath = el.GetElementsByTagName(nameof(ResidualBlock.MainPath));
        var skipPath = el.GetElementsByTagName(nameof(ResidualBlock.ResidualPath));

        INetworkModule? main = mainPath.Count > 0 ? Deserialize((XmlElement?)mainPath[0]) : null;
        INetworkModule? skip = skipPath.Count > 0 ? Deserialize((XmlElement?)skipPath[0]) : null;

        if (main is null)
            throw new NullReferenceException(nameof(ResidualBlock.MainPath));

        var resType = el.GetAttribute("Type");
        return resType switch
        {
            nameof(ResidualAdd) => new ResidualAdd(main, skip),
            nameof(ResidualConcat) => new ResidualConcat(main, skip),

            _ => throw new NotSupportedException(resType)
        };
    }

    public XmlElement Visit(SequentialBlock block, XmlDocument doc)
    {
        var elem = doc.CreateElement(nameof(SequentialBlock));

        for (var l = 0; l < block.LayerCount; l++)
        {
            var layer = block.GetLayer(l);
            if (layer is not IBlockVisitable visitable)
                continue;

            var subElem = visitable.Accept(this, doc);
            elem.AppendChild(subElem);
        }

        return elem;
    }
    public INetworkModule DecodeSequentialBlock(XmlElement el)
    {
        SequentialBlock block = new SequentialBlock();
        foreach (XmlElement child in el.ChildNodes)
        {
            var node = Deserialize(child);
            block.Add(node);
        }
        return block;
    }
}