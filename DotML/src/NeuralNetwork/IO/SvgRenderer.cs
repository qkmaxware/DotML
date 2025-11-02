
namespace DotML.Network.IO;

/// <summary>
/// A visitor which can compute the number of trainable and un-trainable parameters in a network.
/// </summary>
public class SvgRenderer
{
    private const int Padding = 10; // Padding for each side
    private const int LayerWidth = 64;
    private const int LayerHeight = 128;
    private const int PaddedLayerWidth = LayerWidth + Padding * 2;
    private const int PaddedLayerHeight = LayerHeight + Padding * 2;

    private const string ArrowMarkerId = "Arrow";
    private const string GridPatternId = "Grid";

    private void WriteHeader(int width, int height, TextWriter writer)
    {
        writer.WriteLine($"<svg width=\"{width}\" height=\"{height}\" xmlns=\"http://www.w3.org/2000/svg\">");
        writer.WriteLine("<defs>");
        // Arrow marker
        writer.WriteLine(
@$"<marker
    orient='auto'
    refY='0'
    refX='0'
    id='{ArrowMarkerId}'
    style='overflow:visible'>
    <path
        id='path978-6'
        style='fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#000000;stroke-width:0.625;stroke-linejoin:round;stroke-opacity:1'
        d='M 8.7185878,4.0337352 -2.2072895,0.01601326 8.7185884,-4.0017078 c -1.7454984,2.3720609 -1.7354408,5.6174519 -6e-7,8.035443 z'
        transform='scale(-0.6)' />
</marker>");
        // Checkerboard pattern
        writer.WriteLine($"<pattern id='{GridPatternId}' width='10' height='10' patternUnits='userSpaceOnUse'>");
        writer.WriteLine("<rect x='0' y='0' width='10' height='10' fill='white'></rect>");
        writer.WriteLine("<path d='M 10 0 L 0 0 0 10' fill='white' stroke='gray' stroke-width='1'/>");
        writer.WriteLine("</pattern>");
        writer.WriteLine("</defs>");
    }

    private void WriterFooter(TextWriter writer)
    {
        writer.WriteLine($"</svg>");
    }

    private static string GetSvgTemplate(string? name, params object[] args)
    {
        const string defaultTemplate = "DotML.src.NeuralNetwork.IO.SvgTemplates.ActivationLayer.svg.part";
        Stream? stream;
        if (string.IsNullOrEmpty(name))
        {
            stream = typeof(SvgRenderer).Assembly.GetManifestResourceStream(defaultTemplate)!;
        }
        else
        {
            stream = typeof(SvgRenderer).Assembly.GetManifestResourceStream("DotML.src.NeuralNetwork.IO.SvgTemplates." + name);
            if (stream is null)
            {
                stream = typeof(SvgRenderer).Assembly.GetManifestResourceStream(defaultTemplate)!;
            }
        }
        using var reader = new StreamReader(stream);
        var str = reader.ReadToEnd();
        stream.Dispose();

        return args is not null && args.Length > 0 ? string.Format(str, args) : str;
    }

    public string RenderToString(INetworkModule module)
    {
        using (var stringWriter = new StringWriter())
        {
            RenderToStream(module, stringWriter);
            return stringWriter.ToString();
        }
    }

    public void RenderToFile(INetworkModule module, string filePath)
    {
        using (var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write))
        using (var writer = new StreamWriter(stream))
        {
            RenderToStream(module, writer);
        }
    }

    public void RenderToStream(INetworkModule module, TextWriter writer)
    {
        Layout layout;
        if (module is IBlockVisitable visitable)
        {
            var builder = new LayoutBuilder();
            visitable.Accept(builder, new LayoutPosition { TrackIndex = 0, PositionIndex = 0 });
            layout = builder.Layout;
        }
        else
        {
            layout = new Layout();
        }

        // Writer header
        var layoutWidth = layout.Width;
        WriteHeader((layoutWidth + 2) * PaddedLayerWidth, layout.Height * PaddedLayerHeight, writer); // Reserve one space for input, and one for output

        // Write each module
        var svgModuleVisitor = new ModuleToSvgString();
        writer.WriteLine("<g id='modules'>");
        writer.WriteLine($"<g id='input' transform='translate(0,{Padding})'>");
        var shape = module is ArchitectureBlock arch && arch.RequiredInputShape.HasValue
            ? arch.RequiredInputShape.Value.NormalizeRank(3)
            : TensorShape.Scalar;
        writer.WriteLine(GetSvgTemplate("InputTensor.svg.part", shape.LengthOrDefault(0), shape.LengthOrDefault(1), shape.LengthOrDefault(2)));
        writer.WriteLine("</g>");
        for (var trackIndex = 0; trackIndex < layout.Height; trackIndex++)
        {
            writer.WriteLine("<g id='track-" + trackIndex + "'>");
            var track = layout.AddOrGetTrack(trackIndex);
            for (var positionIndex = 0; positionIndex < track.Width; positionIndex++)
            {
                var item = track.GetItem(positionIndex);
                var x = (positionIndex + 1) * PaddedLayerWidth + Padding; // Reserve one space for input
                var y = trackIndex * PaddedLayerHeight + Padding;

                writer.WriteLine($"<g id='module-{trackIndex}-{positionIndex}' transform='translate({x},{y})'>");
                if (item.Module is not null)
                {
                    if (item.Module is IBlockVisitable visitableModule)
                    {
                        var svg = visitableModule.Accept(svgModuleVisitor, None.Value);
                        writer.WriteLine(svg);
                    }
                    else
                    {
                        var svg = GetSvgTemplate(null);
                        writer.WriteLine(svg);
                    }
                }
                else if (item.Combiner.HasValue)
                {
                    var combine = item.Combiner.Value;
                    var actionStringLength = (combine.Action?.Length ?? 0) + 1;

                    // Draw combiner connections
                    var halfH = LayerHeight / 2;
                    var y1 = combine.FirstTrackIndex * PaddedLayerHeight + Padding + halfH;
                    var y2 = combine.SecondTrackIndex * PaddedLayerHeight + Padding + halfH;
                    writer.WriteLine($"<line x1='0' y1='{halfH}' x2='{LayerWidth - actionStringLength}' y2='{y1 - y}' stroke='black' stroke-width='2' marker-end='url(#{ArrowMarkerId})' />");
                    writer.WriteLine($"<line x1='0' y1='{halfH}' x2='{LayerWidth - actionStringLength}' y2='{y2 - y}' stroke='black' stroke-width='2' marker-end='url(#{ArrowMarkerId})' />");

                    // Draw combiner symbol
                    writer.WriteLine($"<text x='{LayerWidth - actionStringLength + 1}' y='{halfH}' text-anchor='middle' dominant-baseline='middle''>{combine.Action}</text>");
                }
                else if (item.Splitter.HasValue)
                {
                    var split = item.Splitter.Value;

                    // Draw splitter connections
                    var halfH = LayerHeight / 2;
                    var y1 = split.FirstTrackIndex * PaddedLayerHeight + Padding + halfH;
                    var y2 = split.SecondTrackIndex * PaddedLayerHeight + Padding + halfH;
                    writer.WriteLine($"<line x1='0' y1='{halfH}' x2='0' y2='{y1 - y}' stroke='black' stroke-width='2' marker-end='url(#{ArrowMarkerId})' />");
                    writer.WriteLine($"<line x1='0' y1='{halfH}' x2='0' y2='{y2 - y}' stroke='black' stroke-width='2' marker-end='url(#{ArrowMarkerId})' />");

                    // Draw splitter labels
                    // TODO 
                }
                else if (item.Unknown is true)
                {
                    var svg = GetSvgTemplate(null);
                    writer.WriteLine(svg);
                }
                writer.Write("</g>\n");
            }
            writer.WriteLine("</g>");
        }
        writer.WriteLine($"<g id='output' transform='translate({(layoutWidth + 1) * PaddedLayerWidth},{Padding})'>");
        writer.WriteLine(GetSvgTemplate("ActivationLayer.svg.part", "Output"));
        writer.WriteLine("</g>");
        writer.WriteLine("</g>");

        // Write footer
        WriterFooter(writer);
    }


    struct Split
    {
        public string? FirstTrackLabel;
        public int FirstTrackIndex;
        public string? SecondTrackLabel;
        public int SecondTrackIndex;
    }
    struct Combine
    {
        public string? Action;
        public int FirstTrackIndex;
        public int SecondTrackIndex;
    }
    struct LayoutItem
    {
        // Disjoint union of several options
        public INetworkModule? Module;  // Use this if it's a normal module
        public Combine? Combiner;        // Use this if it's a combiner node
        public Split? Splitter;        // Use this if it's a splitter node
        public bool? Unknown;          // Use this if it's an unknown module
    }

    private class LayoutTrack
    {
        private List<LayoutItem> modules = new List<LayoutItem>();
        public int Width => modules.Count;
        public LayoutItem GetItem(int index)
        {
            if (index >= modules.Count)
            {
                return new LayoutItem();
            }

            return modules[index];
        }
        public void SetModule(int index, INetworkModule module)
        {
            if (index >= modules.Count)
            {
                while (modules.Count <= index)
                    modules.Add(new LayoutItem());
            }

            modules[index] = new LayoutItem { Module = module };
        }
        public void SetItem(int index, LayoutItem item)
        {
            if (index >= modules.Count)
            {
                while (modules.Count <= index)
                    modules.Add(new LayoutItem());
            }

            modules[index] = item;
        }
    }

    private class Layout
    {
        private List<LayoutTrack> tracks = new List<LayoutTrack>();
        public int Height => tracks.Count;
        public int Width => tracks.Max(t => t.Width);
        public LayoutTrack AddOrGetTrack(int index)
        {
            if (index >= tracks.Count)
            {
                while (tracks.Count <= index)
                    tracks.Add(new LayoutTrack());
            }

            return tracks[index];
        }
        public void SetModule(int trackIndex, int positionIndex, INetworkModule module)
        {
            var track = AddOrGetTrack(trackIndex);
            track.SetModule(positionIndex, module);
        }
        public void SetItem(int trackIndex, int positionIndex, LayoutItem item)
        {
            var track = AddOrGetTrack(trackIndex);
            track.SetItem(positionIndex, item);
        }
    }

    private struct LayoutPosition
    {
        public int TrackIndex;
        public int PositionIndex;
    }

    private class LayoutBuilder : IBlockVisitor<LayoutPosition, LayoutPosition>
    {
        public Layout Layout { get; } = new Layout();

        public LayoutPosition Visit(object? obj, LayoutPosition arg)
        {
            Layout.SetItem(arg.TrackIndex, arg.PositionIndex, new LayoutItem { Unknown = true });
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(Activation activation, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, activation);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(SoftmaxOutput softmax, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, softmax);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(Conv2D conv, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, conv);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(TransposeConv2D tconv, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, tconv);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(DenseLinear dense, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, dense);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(Dropout dropout, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, dropout);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(BatchNorm2D norm, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, norm);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(GroupNorm norm, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, norm);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(LayerNorm norm, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, norm);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(PixelShuffler shuffle, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, shuffle);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(AvgPool2D pool, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, pool);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(MaxPool2D pool, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, pool);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(MinPool2D pool, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, pool);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(GlobalAvgPool2D pool, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, pool);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(GlobalMaxPool2D pool, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, pool);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(GlobalMinPool2D pool, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, pool);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(Reshape reshape, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, reshape);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(Flatten flatten, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, flatten);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(Center2D center, LayoutPosition arg)
        {
            Layout.SetModule(arg.TrackIndex, arg.PositionIndex, center);
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = arg.PositionIndex + 1 };
        }

        public LayoutPosition Visit(ResidualBlock block, LayoutPosition arg)
        {
            // Do splitter
            Layout.SetItem(arg.TrackIndex, arg.PositionIndex, new LayoutItem { Splitter = new Split
            {
                FirstTrackLabel = "Main",
                FirstTrackIndex = arg.TrackIndex,
                SecondTrackLabel = "Skip",
                SecondTrackIndex = arg.TrackIndex + 1
            } });

            // Do the main track
            if (block.MainPath is IBlockVisitable visitable)
                visitable.Accept(this, new LayoutPosition
                {
                    TrackIndex = arg.TrackIndex,
                    PositionIndex = arg.PositionIndex + 1
                });

            // Do the skip track
            if (block.ResidualPath is IBlockVisitable skipVisitable)
                skipVisitable.Accept(this, new LayoutPosition
                {
                    TrackIndex = arg.TrackIndex + 1,
                    PositionIndex = arg.PositionIndex + 1
                });

            // Do the combiner
            var eol = Math.Max(Layout.AddOrGetTrack(arg.TrackIndex).Width, Layout.AddOrGetTrack(arg.TrackIndex + 1).Width);
            Layout.SetItem(arg.TrackIndex, eol, new LayoutItem { Combiner = new Combine
            {
                Action = block switch
                {
                    ResidualAdd => "+",
                    ResidualConcat => "|",
                    _ => "fn"
                },
                FirstTrackIndex = arg.TrackIndex,
                SecondTrackIndex = arg.TrackIndex + 1
            } });

            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = eol + 1 };
        }

        public LayoutPosition Visit(SequentialBlock block, LayoutPosition arg)
        {
            var eol = arg.PositionIndex;
            var nextPos = arg;
            for (int i = 0; i < block.LayerCount; i++)
            {
                var module = block.GetLayer(i);
                if (module is IBlockVisitable visitable)
                {
                    nextPos = visitable.Accept(this, nextPos);
                }
            }
            return new LayoutPosition { TrackIndex = arg.TrackIndex, PositionIndex = nextPos.PositionIndex };
        }
    }

    private class ModuleToSvgString : IBlockVisitor<string>
    {
        public string Visit(object? obj, None args)
        {
            return GetSvgTemplate(null, obj?.GetType()?.Name ?? "Unknown");
        }

        public string Visit(Activation activation, None arg)
        {
            return GetSvgTemplate("ActivationLayer.svg.part", activation.ActivationFunction.GetType().Name);
        }

        public string Visit(SoftmaxOutput softmax, None arg)
        {
            return GetSvgTemplate("ActivationLayer.svg.part", "Softmax");
        }

        public string Visit(Conv2D conv, None arg)
        {
            return GetSvgTemplate("ConvolutionLayer.svg.part");
        }

        public string Visit(TransposeConv2D tconv, None arg)
        {
            return GetSvgTemplate("ConvolutionLayer.svg.part");
        }

        public string Visit(DenseLinear dense, None arg)
        {
            return GetSvgTemplate("DenseLinearLayer.svg.part");
        }

        public string Visit(Dropout dropout, None arg)
        {
            return GetSvgTemplate("DropoutLayer.svg.part");
        }

        public string Visit(BatchNorm2D norm, None arg)
        {
            return GetSvgTemplate("Normalize.svg.part", nameof(BatchNorm2D));
        }

        public string Visit(GroupNorm norm, None arg)
        {
            return GetSvgTemplate("Normalize.svg.part", nameof(GroupNorm));
        }

        public string Visit(LayerNorm norm, None arg)
        {
            return GetSvgTemplate("Normalize.svg.part", nameof(LayerNorm));
        }

        public string Visit(PixelShuffler shuffle, None arg)
        {
            return GetSvgTemplate("PixelShuffle.svg.part");
        }

        public string Visit(AvgPool2D pool, None arg)
        {
            return GetSvgTemplate("Pooling.svg.part");
        }

        public string Visit(MaxPool2D pool, None arg)
        {
            return GetSvgTemplate("Pooling.svg.part");
        }

        public string Visit(MinPool2D pool, None arg)
        {
            return GetSvgTemplate("Pooling.svg.part");
        }

        public string Visit(GlobalAvgPool2D pool, None arg)
        {
            return GetSvgTemplate("Pooling.svg.part");
        }

        public string Visit(GlobalMaxPool2D pool, None arg)
        {
            return GetSvgTemplate("Pooling.svg.part");
        }

        public string Visit(GlobalMinPool2D pool, None arg)
        {
            return GetSvgTemplate("Pooling.svg.part");
        }

        public string Visit(Reshape reshape, None arg)
        {
            throw new NotImplementedException();
        }

        public string Visit(Flatten flatten, None arg)
        {
            return GetSvgTemplate("FlatteningLayer.svg.part");
        }

        public string Visit(Center2D center, None arg)
        {
            return GetSvgTemplate(null);
        }

        public string Visit(ResidualBlock block, None arg)
        {
            return string.Empty;
        }

        public string Visit(SequentialBlock block, None arg)
        {
            return string.Empty;
        }
    }
}