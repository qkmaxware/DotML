using System.Drawing;

namespace DotML.Network;

/// <summary>
/// Writer to encode layer information to a text format
/// </summary>
public class SvgWriter : ILayerInputVisitor<SvgWriter.SvgBuilder> {
    private int PaddingTop = 10;
    private int PaddingBottom = 10;
    private int LayerWidth = 64;
    private int LayerHeight = 128;

    public struct SvgBuilder {
        public FeedforwardNetwork Network {get; private set;}
        public TextWriter Writer {get; private set;}
        public int LayerIndex {get; set;}
        public Rectangle Viewport {get; set;}
        private int _skip_depth;
        public int SkipDepth {
            get => _skip_depth;
            set => _skip_depth = Math.Max(0, value);
        }

        public SvgBuilder(FeedforwardNetwork network, TextWriter writer) {
            this.Network = network;
            this.Writer = writer;
        }
    }

    private string get_template(string name) {
        using var stream = typeof(SvgWriter).Assembly.GetManifestResourceStream(name);
        if (stream == null)
            return string.Empty;
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    public string Write(FeedforwardNetwork network) {
        using var writer = new StringWriter();
        WriteTo(network, writer);
        return writer.ToString();
    }

    public void WriteTo(FeedforwardNetwork network, TextWriter writer) {
        var layers = network.LayerCount;
        var width = (layers + 2) * LayerWidth;
        var PaddingBottom = this.PaddingBottom + 3 * Enumerable.Range(0, network.LayerCount).Select((ind) => network.GetLayer(ind)).Count();
        var height = LayerHeight + PaddingTop + PaddingBottom;

        writer.WriteLine($"<svg width=\"{width}\" height=\"{height}\" xmlns=\"http://www.w3.org/2000/svg\">");
        writer.WriteLine("<defs>");
        // Arrow marker
        writer.WriteLine(
@"<marker
    orient='auto'
    refY='0'
    refX='0'
    id='Arrow'
    style='overflow:visible'>
    <path
        id='path978-6'
        style='fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#000000;stroke-width:0.625;stroke-linejoin:round;stroke-opacity:1'
        d='M 8.7185878,4.0337352 -2.2072895,0.01601326 8.7185884,-4.0017078 c -1.7454984,2.3720609 -1.7354408,5.6174519 -6e-7,8.035443 z'
        transform='scale(-0.6)' />
</marker>");
        // Checkerboard pattern
        writer.WriteLine("<pattern id='Grid' width='10' height='10' patternUnits='userSpaceOnUse'>");
            writer.WriteLine("<rect x='0' y='0' width='10' height='10' fill='white'></rect>");
            writer.WriteLine("<path d='M 10 0 L 0 0 0 10' fill='white' stroke='gray' stroke-width='1'/>");
        writer.WriteLine("</pattern>");
        writer.WriteLine("</defs>");
        var builder = new SvgBuilder(network, writer);

        // Write input layer
        var isize = network.InputShape;
        writer.WriteLine($"<g id='input' transform='translate({0},{PaddingTop})'>");
        writer.WriteLine($"<text x='{LayerWidth >> 1}' y='{-PaddingTop}' text-anchor='middle' dominant-baseline='hanging'>Input</text>"); 
        if (isize.Rows == 1 || isize.Columns == 1) {
            writer.WriteLine(string.Format(
                get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.InputNeurons.svg.part"),
                $"{isize.Channels}x{isize.Rows * isize.Columns}"
            ));
        } else {
            writer.WriteLine(string.Format(
                get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.InputTensor.svg.part"),
                isize.Channels,
                isize.Rows,
                isize.Columns
            ));
        }
        writer.WriteLine($"</g>");

        // Write inner layers
        for (var layerIndex = 0; layerIndex < layers; layerIndex++) {
            var layer = network.GetLayer(layerIndex);
            var rect = new Rectangle(x: (layerIndex + 1) * LayerWidth, y: PaddingTop, width: LayerWidth, height: LayerHeight);
            
            builder.LayerIndex = layerIndex;
            builder.Viewport = rect;
            
            writer.WriteLine($"<g id='layer-{layerIndex}' transform='translate({rect.X},{rect.Y})' data-layer-index='{layerIndex}' data-layer-input='{layer.InputShape}' data-layer-output='{layer.OutputShape}' data-layer-type='{layer.GetType().Name}'>");
                writer.WriteLine($"<text x='{LayerWidth >> 1}' y='{-PaddingTop}' text-anchor='middle' dominant-baseline='hanging'>Layer {layerIndex}</text>");
                layer.Visit(this, builder);
            writer.WriteLine($"</g>");
        }

        // Write output layer
        var osize = network.OutputShape;
        writer.WriteLine($"<g id='output' transform='translate({(layers + 1) * LayerWidth},{PaddingTop})'>");
        writer.WriteLine($"<text x='{LayerWidth >> 1}' y='{-PaddingTop}' text-anchor='middle' dominant-baseline='hanging'>Output</text>"); 
        if (osize.Rows == 1 || osize.Columns == 1) {
            writer.WriteLine(string.Format(
                get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.InputNeurons.svg.part"),
                $"{osize.Channels}x{osize.Rows * osize.Columns}"
            ));
        } else {
            writer.WriteLine(string.Format(
                get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.InputTensor.svg.part"),
                osize.Channels,
                osize.Rows,
                osize.Columns
            ));
        }
        writer.WriteLine($"</g>");

        writer.WriteLine($"</svg>");
    }

    public void Visit(ConvolutionLayer layer, SvgBuilder args) {
        args.Writer.WriteLine(get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.ConvolutionLayer.svg.part"));
    }

    public void Visit(DepthwiseConvolutionLayer layer, SvgBuilder args) {
        args.Writer.WriteLine(get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.DepthwiseConvolutionLayer.svg.part"));
    }

    public void Visit(TransposeConvolutionLayer layer, SvgBuilder args) {
        // Basically the same thing as a convolution layer so use the same diagram
        args.Writer.WriteLine(get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.ConvolutionLayer.svg.part"));
    }

    public void Visit(PixelShuffle layer, SvgBuilder args) {
        args.Writer.WriteLine(get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.PixelShuffle.svg.part"));
    }

    public void Visit(PoolingLayer layer, SvgBuilder args) {
        args.Writer.WriteLine(get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.Pooling.svg.part"));
    }

    public void Visit(FlatteningLayer layer, SvgBuilder args) {
        args.Writer.WriteLine(get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.FlatteningLayer.svg.part"));
    }

    public void Visit(DropoutLayer layer, SvgBuilder args) {
        args.Writer.WriteLine(get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.DropoutLayer.svg.part"));
    }

    public void Visit(INormalizationLayer layer, SvgBuilder args) {
        args.Writer.WriteLine(
            string.Format(
                get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.Normalize.svg.part"),
                layer.GetType().Name
            )
        );
    }

    public void Visit(LayerNorm layer, SvgBuilder args) {
        Visit((INormalizationLayer)layer, args);
    }

    public void Visit(BatchNorm layer, SvgBuilder args) {
        Visit((INormalizationLayer)layer, args);
    }

    public void Visit(DenseLinearLayer layer, SvgBuilder args) {
        args.Writer.WriteLine(get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.DenseLinearLayer.svg.part"));
    }

    public void Visit(ActivationLayer layer, SvgBuilder args) {
        args.Writer.WriteLine(
            string.Format(
                get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.ActivationLayer.svg.part"),
                layer.ActivationFunction.ToString() + "(x)"
            )
        );
    }

    public void Visit(SoftmaxLayer layer, SvgBuilder args) {
        args.Writer.WriteLine(
            string.Format(
                get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.ActivationLayer.svg.part"),
                "Softmax(x)"
            )
        );
    }

    public void Visit(InputCapture capture, SvgBuilder args) {
        args.Writer.WriteLine(
            string.Format(
                get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.Skip.svg.part"),
                string.Empty
            )
        );
        args.SkipDepth++;
    }

    private void Visit(SkipConnection skip, SvgBuilder args, string icon) {
        var text = get_template("DotML.src.NeuralNetwork.IO.SvgTemplates.Skip.svg.part");
        args.Writer.WriteLine(
            string.Format(
                text,
                icon
            )
        );
        args.SkipDepth--;
        // Cx,Cy,R from the template
        var cx = 46.287514;
        var cy = 64; 
        var r = 10.016369;
        var cbottom_start = cy + r;
        var cbottom_end = cy + Math.Ceiling(r);
        var capture = skip.CaptureSource;
        var hline_height = LayerHeight + args.SkipDepth * 3 + 1; // From the number of skips deep in the network TODO
        var layer_index = Enumerable.Range(0, args.Network.LayerCount).Where((ind) => ReferenceEquals(args.Network.GetLayer(ind), capture)).FirstOrDefault(-1);
        if (layer_index == -1)
            return;

        var initial_offset = args.Viewport.X - (LayerWidth * (layer_index + 1) + cx);
        args.Writer.WriteLine(
            $"<path d='M{-initial_offset} {cbottom_start} L{-initial_offset} {hline_height} L{cx} {hline_height} L{cx} {cbottom_end}' style='fill:none;stroke:#000000;stroke-width:0.265;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1;marker-end:url(#Arrow)'/>"
        );
    }

    public void Visit(AdditionSkipConnection skip, SvgBuilder args) {
        Visit(skip, args, "+");
    }

    public void Visit(ConcatenationSkipConnection skip, SvgBuilder args) {
        Visit(skip, args, "||");
    }

}