using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Base class representing a convolutional neural network (CNN)
/// </summary>
public class ConvolutionalFeedforwardNetwork: 
    ILayeredNeuralNetwork<IConvolutionalFeedforwardNetworkLayer>, 
    IDiagrammable, ISafetensorable, IMarkdownable, IHtmlable
{
    private List<IConvolutionalFeedforwardNetworkLayer> layers = new List<IConvolutionalFeedforwardNetworkLayer>();

    /// <summary>
    /// Number of layers
    /// </summary>
    [JsonIgnore] public int LayerCount => layers.Count;
    /// <summary>
    /// Size of the input 
    /// </summary>
    [JsonIgnore] public Shape3D InputShape => GetFirstLayer().InputShape;
    /// <summary>
    /// Size of the output / number of classifications
    /// </summary>
    [JsonIgnore] public Shape3D OutputShape => GetOutputLayer().OutputShape;

    /// <summary>
    /// Reference to the first layer in the network
    /// </summary>
    /// <returns>layer</returns>
    public IConvolutionalFeedforwardNetworkLayer GetFirstLayer() => layers[0];

    /// <summary>
    /// Get a specific layer by index
    /// </summary>
    /// <param name="index">index of layer</param>
    /// <returns>layer</returns>
    public IConvolutionalFeedforwardNetworkLayer GetLayer(int index) => layers[index];

    /// <summary>
    /// Reference to the output layer of the network
    /// </summary>
    /// <returns>layer</returns>
    public IConvolutionalFeedforwardNetworkLayer GetOutputLayer() => layers[^1];


    public ConvolutionalFeedforwardNetwork(params IConvolutionalFeedforwardNetworkLayer[] layers) {
        this.layers.AddRange(layers);
    }

    public ConvolutionalFeedforwardNetwork(IEnumerable<IConvolutionalFeedforwardNetworkLayer> layers) {
        this.layers.AddRange(layers);
    }

    public void Initialize(IInitializer initializer) {
        foreach (var layer in this.layers) {
            layer.Initialize(initializer);
        }
    }

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public int TrainableParameterCount() => this.layers.Select(layer => layer.TrainableParameterCount()).Sum();

    public void ValidateSizes() {
        var input_size = this.InputShape;
        int layer_index = 0;
        foreach (var layer in this.layers) { 
            if (input_size != layer.InputShape) {
                throw new ArgumentException($"Layer {layer_index} expects an input shape of {layer.InputShape} but is receiving an input of shape {input_size} from the previous layer.");
            }
            input_size = layer.OutputShape;
            layer_index++;
        }
    }

    public Vec<double> PredictSync(Matrix<double>[] values) {
        var ishape = this.InputShape;

        if (values.Length != ishape.Channels) {
            throw new ArgumentException($"Invalid number of channels for input. Expected {ishape.Channels}, got {values.Length}.");
        }
        foreach (var matrix in values) {
            if (matrix.Columns != ishape.Columns)
                throw new ArgumentException($"Invalid channel width. Expected {ishape.Columns}, got {matrix.Columns}.");
            if (matrix.Rows != ishape.Rows)
                throw new ArgumentException($"Invalid channel height. Expected {ishape.Rows}, got {matrix.Rows}.");
        }

        Matrix<double>[] input = values;
        var layer_index = 0;
        foreach (var layer in this.layers) {
            if (!layer.DoesShapeMatchInputShape(input))
                throw new ArithmeticException($"Input of shape {input.Length}x{input.FirstOrDefault().Rows}x{input.FirstOrDefault().Columns} is incompatible with layer {layer_index} input's of shape {layer.InputShape}.");
            input = layer.EvaluateSync(input);
            layer_index++;
        }
        return Vec<double>.Wrap(input[0].FlattenRows().ToArray());
    }

    public Vec<double> PredictSync(Vec<double> input) {
        var ishape = this.InputShape;
        return PredictSync(input.Shape(
            new Shape2D(ishape.Rows, ishape.Columns), 
                ishape.Channels
            ).ToArray()
        ); 
    }

    /// <summary>
    /// Load the weights of the network from the given safetensors
    /// </summary>
    /// <param name="sb">safetensors</param>
    public void FromSafetensor(Safetensors sb) {
        var reader = new LayerSafetensorReader(sb);
        for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
            var layer = this.GetLayer(layerIndex);
            if (!layer.Visit(reader, layerIndex)) {
                throw new ArgumentException($"Failed to load all tensors for layer {layerIndex}.");
            }
        }
    }

    /// <summary>
    /// Output this network's configuration in the safetensor format
    /// </summary>
    public Safetensors ToSafetensor() {
        var writer = new LayerSafetensorWriter();

        for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
            var layer = this.GetLayer(layerIndex);
            if (!layer.Visit(writer, layerIndex)) {
                throw new ArgumentException($"Failed to save all tensors for layer {layerIndex}.");
            }
        }

        return writer.ToSafetensors();
    }

    /// <summary>
    /// Output this network's configuration in the safetensor format
    /// </summary>
    /// <param name="writer">binary writer to write to</param>
    public void ToSafetensor(BinaryWriter writer) {
        var sb = ToSafetensor();
        sb.WriteTo(writer);
    }

    public string ToSvg() {
        // Draw input layer
        StringBuilder s = new StringBuilder();

        var max_outputs_matrices = this.InputShape.Channels;
        var kernel_buffer = 0;
        var last_output_matrices = 0;
        var kernel_offset = 6;
        foreach (var layer in this.layers) {
            switch (layer) {
                case ConvolutionLayer convo:
                    var convo_outputs = convo.FilterCount;
                    max_outputs_matrices = Math.Max(max_outputs_matrices, convo_outputs);
                    last_output_matrices = convo_outputs;
                    kernel_buffer = Math.Max(kernel_buffer, convo.Filters.Select(x => x.Count).Max() * kernel_offset);
                    break;
                case PoolingLayer pool:
                    break;
                case FullyConnectedLayer connect:
                    max_outputs_matrices = Math.Max(max_outputs_matrices, connect.OutputShape.Count);
                    last_output_matrices = 1;
                    break;
            }
        }


        var matrix_size = 80;
        var neuron_radius = 25;
        var layer_width = 125;
        var header_size = 32 + kernel_buffer;
        var footer_size = 32;
        var layer_height = header_size + max_outputs_matrices * matrix_size + footer_size;
        var layer_buffer = 32;

        var img_height = layer_height;
        var img_width = (layer_buffer + layer_width) * (this.LayerCount + 1);

        var units = "px";
        s.AppendLine($"<svg width='{img_width}{units}' height='{img_height}{units}' xmlns='http://www.w3.org/2000/svg'>");
        s.AppendLine("<defs>");
            s.AppendLine("<pattern id='grid' width='12' height='12' patternUnits='userSpaceOnUse'>");
                s.AppendLine("<rect x='0' y='0' width='12' height='12' fill='white'></rect>");
                s.AppendLine("<path d='M 12 0 L 0 0 0 12' fill='white' stroke='gray' stroke-width='1'/>");
            s.AppendLine("</pattern>");
        s.AppendLine("</defs>");

        // Draw "input" layer
        s.AppendLine($"<text x='{layer_width/2}' y='{16}' text-anchor='middle'>Input</text>");
        var input_count = this.InputShape.Channels;
        var matrix_offset = (layer_width - matrix_size) / 2;
        for (var i = 0; i < input_count; i++) {
            s.AppendLine($"<rect x='{matrix_offset}' y='{i * matrix_size + header_size}' width='{matrix_size}' height='{matrix_size}' fill='url(#grid)' stroke='black'></rect>");
        }
        var start_layer_midpoint_y = header_size + (input_count * matrix_size) / 2;

        // Draw all the rest of the layers
        last_output_matrices = input_count;
        var last_layer_midpoint_y = start_layer_midpoint_y;
        bool last_layer_was_fully_connected = false;
        for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
            s.AppendLine($"<g id='layer{layerIndex}'>");
            // Compute dimensions
            var layer = this.GetLayer(layerIndex);
            var layer_start_x   = (layer_buffer + layer_width) * (layerIndex + 1);
            var layer_start_buffer_x = layer_start_x - layer_buffer;
            var layer_end_x     = (layer_buffer + layer_width) * (layerIndex + 2);
            var layer_start_y   = 0;
            var layer_end_y     = layer_height;
            var layer_midpoint_y = header_size + (layer_height) / 2;
        
            // Draw layer specifics
            switch (layer) {
                case ConvolutionLayer convo:
                    for (var filterIdx = convo.FilterCount; filterIdx > 0; filterIdx--) {
                        var filterIndex = filterIdx - 1;
                        var filter = convo.Filters[filterIndex];
                        for (var kernel = filter.Count; kernel > 0; kernel--) {
                            var kernelIndex = kernel - 1;
                            s.AppendLine($"<rect x='{matrix_offset + layer_start_x + kernelIndex*kernel_offset}' y='{header_size + filterIndex * matrix_size - kernelIndex*kernel_offset}' width='{matrix_size}' height='{matrix_size}' fill='url(#grid)' stroke='black'></rect>");
                        }
                    }
                    layer_midpoint_y = header_size + (convo.FilterCount * matrix_size) / 2;
                    break;
                case PoolingLayer pool:
                    for (var i = 0; i < last_output_matrices; i++) {
                        s.AppendLine($"<rect x='{matrix_offset + layer_start_x}' y='{header_size + i * matrix_size}' width='{matrix_size}' height='{matrix_size}' fill='url(#grid)' stroke='black'></rect>");
                    }
                    layer_midpoint_y = last_layer_midpoint_y;
                    break;
                case ActivationLayer active:
                    {
                        s.AppendLine("<g id='synapses'>");
                        if (last_layer_was_fully_connected) {
                            for (var i = 0; i < active.InputShape.Count; i++) {
                                var neuron_offset = (matrix_size - 2*neuron_radius) / 2;
                                var center_x = matrix_offset + layer_start_x + matrix_size / 2;
                                var center_y = header_size + i * matrix_size + matrix_size / 2;
                                var in_center_x = (layer_buffer + layer_width) * layerIndex + matrix_offset + matrix_size - neuron_offset; // from the prev_layer
                                var in_center_y = header_size + i * matrix_size + matrix_size / 2;
                                s.AppendLine($"<line x1='{in_center_x}' y1='{in_center_y}' x2='{center_x - neuron_radius}' y2='{center_y}' stroke='gray'/>");
                            }
                        }
                        s.AppendLine("</g>");
                        for (var i = 0; i < active.InputShape.Count; i++) {
                            var center_x = matrix_offset + layer_start_x + matrix_size / 2;
                            var center_y = header_size + i * matrix_size + matrix_size / 2;
                            s.AppendLine($"<circle cx='{center_x}' cy='{center_y}' r='{neuron_radius}' fill='black' stroke='black'/>");
                            s.AppendLine($"<text x='{center_x}' y='{center_y}' text-anchor='middle' fill='white'>F(x)</text>");
                        }
                        layer_midpoint_y = last_layer_midpoint_y;
                    }
                    break;
                case FullyConnectedLayer connect:
                    for (var i = 0; i < connect.OutputShape.Count; i++) {
                        var neuron_offset = (matrix_size - 2*neuron_radius) / 2;
                        var center_x = matrix_offset + layer_start_x + matrix_size / 2;
                        var center_y = header_size + i * matrix_size + matrix_size / 2;
                        s.AppendLine("<g id='synapses'>");
                        if (last_layer_was_fully_connected) {
                            for (var j = 0; j < connect.InputShape.Count; j++) {
                                var in_center_x = (layer_buffer + layer_width) * layerIndex + matrix_offset + matrix_size - neuron_offset; // from the prev_layer
                                var in_center_y = header_size + j * matrix_size + matrix_size / 2;
                                s.AppendLine($"<line x1='{in_center_x}' y1='{in_center_y}' x2='{center_x - neuron_radius}' y2='{center_y}' stroke='gray'/>");
                            }
                        }
                        s.AppendLine("</g>");
                        s.AppendLine($"<circle cx='{center_x}' cy='{center_y}' r='{neuron_radius}'/>");
                    }
                    layer_midpoint_y = header_size + (connect.OutputShape.Count * matrix_size) / 2;
                    break;
                case SoftmaxLayer softmax:
                    s.AppendLine($"<text x='{layer_width/2}' y='{layer_midpoint_y}' text-anchor='middle'>softmax(x)</text>");
                    break;
            }

            // Draw dotted arrow
            if (!last_layer_was_fully_connected) {
                s.AppendLine($"<line x1='{layer_start_buffer_x}' y1='{last_layer_midpoint_y}' x2='{layer_start_x}' y2='{layer_midpoint_y}' stroke-dasharray='4' stroke='gray'/>");
            }
            // Draw title
            s.AppendLine($"<text x='{layer_start_x + layer_width/2}' y='{layer_start_y + 16}' text-anchor='middle'>{layer.GetType().Name}</text>");


            switch (layer) {
                case ConvolutionLayer convo:
                    last_output_matrices = convo.FilterCount;
                    last_layer_was_fully_connected = false;
                    break;
                case PoolingLayer pool:
                    layer_midpoint_y = last_layer_midpoint_y;
                    last_layer_was_fully_connected = false;
                    break;
                case FullyConnectedLayer connect:
                    last_output_matrices = 1;
                    last_layer_was_fully_connected = true;
                    break;
                case SoftmaxLayer:
                    last_output_matrices = 1;
                    last_layer_was_fully_connected = false;
                    break;
            }
            last_layer_midpoint_y = layer_midpoint_y;
            s.AppendLine("</g>");
        }

        s.Append("</svg>");
        return s.ToString();
    }

    /// <summary>
    /// Convert this object to a Markdown representation
    /// </summary>
    /// <returns>Markdown serialized string</returns>
    public string ToMarkdown() {
        using StringWriter sb = new StringWriter();

        var md = new LayerMarkdownWriter(sb);
        for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
            var layer = this.GetLayer(layerIndex);
            if (!layer.Visit(md, layerIndex)) {
                throw new ArgumentException($"Failed to print information for layer {layerIndex}.");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Convert this object to an HTML representation
    /// </summary>
    /// <returns>HTML serialized string</returns>
    public string ToHtml() {
        using StringWriter sb = new StringWriter();

        var html = new LayerHtmlWriter(sb);
        for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
            var layer = this.GetLayer(layerIndex);
            if (!layer.Visit(html, layerIndex)) {
                throw new ArgumentException($"Failed to print information for layer {layerIndex}.");
            }
        }

        return sb.ToString();
    }
}