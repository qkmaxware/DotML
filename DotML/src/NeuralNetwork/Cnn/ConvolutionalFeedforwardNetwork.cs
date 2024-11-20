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
    IDiagrammable, ISafetensorable
{
    private List<IConvolutionalFeedforwardNetworkLayer> layers = new List<IConvolutionalFeedforwardNetworkLayer>();

    /// <summary>
    /// Number of layers
    /// </summary>
    [JsonIgnore] public int LayerCount => layers.Count;
    /// <summary>
    /// Size of the input 
    /// </summary>
    [JsonIgnore] public int InputCount => InputImageWidth * InputImageHeight * InputImageChannels;
    /// <summary>
    /// Size of the output / number of classifications
    /// </summary>
    [JsonIgnore] public int OutputCount => GetOutputLayer().OutputCount;

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

    /// <summary>
    /// Allowed width of the input images
    /// </summary>
    public int InputImageWidth {get; init;}
    /// <summary>
    /// Allowed height of the input images
    /// </summary>
    public int InputImageHeight {get; init;}
    /// <summary>
    /// Allowed number of channels in the input images. Commonly 1 for grayscale or 3 for RGB.
    /// </summary>
    public int InputImageChannels {get; init;}

    public ConvolutionalFeedforwardNetwork(int input_width, int input_height, int input_channels, params IConvolutionalFeedforwardNetworkLayer[] layers) {
        this.InputImageWidth = input_width;
        this.InputImageHeight = input_height;
        this.InputImageChannels = input_channels;
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

    public Vec<double> PredictSync(Matrix<double>[] values) {
        if (values.Length != InputImageChannels) {
            throw new ArgumentException($"Invalid number of channels for input. Expected {InputImageChannels}, got {values.Length}.");
        }
        foreach (var matrix in values) {
            if (matrix.Columns != InputImageWidth)
                throw new ArgumentException($"Invalid channel width. Expected {InputImageWidth}, got {matrix.Columns}.");
            if (matrix.Rows != InputImageHeight)
                throw new ArgumentException($"Invalid channel height. Expected {InputImageHeight}, got {matrix.Rows}.");
        }

        Matrix<double>[] input = values;
        foreach (var layer in this.layers) {
            input = layer.EvaluateSync(input);
        }
        return Vec<double>.Wrap(input[0].FlattenRows().ToArray());
    }

    public Vec<double> PredictSync(Vec<double> input) {
        return PredictSync(input.Shape(
            new Shape(this.InputImageHeight, this.InputImageWidth), 
                this.InputImageChannels
            ).ToArray()
        ); 
    }

    /// <summary>
    /// Load the weights of the network from the given safetensors
    /// </summary>
    /// <param name="sb">safetensors</param>
    public void FromSafetensor(SafetensorBuilder sb) {
        for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
            var layer = this.GetLayer(layerIndex);
            switch (layer) {
                case ConvolutionLayer convo:  
                    for (var filterIndex = 0; filterIndex < convo.FilterCount; filterIndex++) {
                        var filter = convo.Filters[filterIndex];
                        for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                            var key = $"Layers[{layerIndex}].Filters[{filterIndex}].Kernel[{kernelIndex}]";
                            if (sb.ContainsKey(key)) {
                                filter[kernelIndex] = sb.GetTensor<double>(key);
                            }
                        }
                    }
                    break;
                case FullyConnectedLayer conn:
                    var wkey = $"Layers[{layerIndex}].Weights";
                    if (sb.ContainsKey(wkey)) {
                        conn.Weights = sb.GetTensor<double>(wkey);
                    }
                    var bkey = $"Layers[{layerIndex}].Biases";
                    if (sb.ContainsKey(bkey)) {
                        conn.Biases = Vec<double>.Wrap(sb.GetTensor<double>(bkey).FlattenRows().ToArray());
                    }
                    break;
            }
        }
    }

    /// <summary>
    /// Output this network's configuration in the safetensor format
    /// </summary>
    /// <param name="writer">binary writer to write to</param>
    public void ToSafetensor(BinaryWriter writer) {
        SafetensorBuilder sb = new SafetensorBuilder();

        for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
            var layer = this.GetLayer(layerIndex);
            switch (layer) {
                case ConvolutionLayer convo:  
                    for (var filterIndex = 0; filterIndex < convo.FilterCount; filterIndex++) {
                        var filter = convo.Filters[filterIndex];
                        for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                            var kernel = filter[kernelIndex];
                            sb.Add($"Layers[{layerIndex}].Filters[{filterIndex}].Kernel[{kernelIndex}]", kernel);
                        }
                    }
                    break;
                case FullyConnectedLayer conn:
                    sb.Add($"Layers[{layerIndex}].Weights", conn.Weights);
                    sb.Add($"Layers[{layerIndex}].Biases", conn.Biases);
                    break;
            }
        }

        sb.WriteTo(writer);
    }

    public string ToSvg() {
        // Draw input layer
        StringBuilder s = new StringBuilder();

        var max_outputs_matrices = this.InputImageChannels;
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
                    max_outputs_matrices = Math.Max(max_outputs_matrices, connect.OutputCount);
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
        var input_count = this.InputImageChannels;
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
                case FullyConnectedLayer connect:
                    for (var i = 0; i < connect.OutputCount; i++) {
                        var neuron_offset = (matrix_size - 2*neuron_radius) / 2;
                        var center_x = matrix_offset + layer_start_x + matrix_size / 2;
                        var center_y = header_size + i * matrix_size + matrix_size / 2;
                        s.AppendLine("<g id='synapses'>");
                        if (last_layer_was_fully_connected) {
                            for (var j = 0; j < connect.InputCount; j++) {
                                var in_center_x = (layer_buffer + layer_width) * layerIndex + matrix_offset + matrix_size - neuron_offset; // from the prev_layer
                                var in_center_y = header_size + j * matrix_size + matrix_size / 2;
                                s.AppendLine($"<line x1='{in_center_x}' y1='{in_center_y}' x2='{center_x - neuron_radius}' y2='{center_y}' stroke='gray'/>");
                            }
                        }
                        s.AppendLine("</g>");
                        s.AppendLine($"<circle cx='{center_x}' cy='{center_y}' r='{neuron_radius}'/>");
                    }
                    layer_midpoint_y = header_size + (connect.OutputCount * matrix_size) / 2;
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
        }

        s.Append("</svg>");
        return s.ToString();
    }
}