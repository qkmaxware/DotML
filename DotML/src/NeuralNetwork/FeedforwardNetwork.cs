using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Base class representing a convolutional neural network (CNN)
/// </summary>
public class FeedforwardNetwork: 
    INamedNetwork,
    IHasStorage,
    ILayeredNeuralNetwork<IFeedforwardNetworkLayer>, 
    IDiagrammable, ISafetensorable, IMarkdownable, IHtmlable
{
    private List<IFeedforwardNetworkLayer> layers = new List<IFeedforwardNetworkLayer>();

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
    public IFeedforwardNetworkLayer GetFirstLayer() => layers[0];

    /// <summary>
    /// Get a specific layer by index
    /// </summary>
    /// <param name="index">index of layer</param>
    /// <returns>layer</returns>
    public IFeedforwardNetworkLayer GetLayer(int index) => layers[index];

    /// <summary>
    /// Reference to the output layer of the network
    /// </summary>
    /// <returns>layer</returns>
    public IFeedforwardNetworkLayer GetOutputLayer() => layers[^1];


    public FeedforwardNetwork(params IFeedforwardNetworkLayer[] layers) {
        this.layers.AddRange(layers);
    }

    public FeedforwardNetwork(IEnumerable<IFeedforwardNetworkLayer> layers) {
        this.layers.AddRange(layers);
    }

    /// <summary>
    /// Remove the layer at the given index
    /// </summary>
    /// <param name="index">index of the layer to delete</param>
    /// <returns>removed layer</returns>
    /// <exception cref="ArgumentException">thrown if the shape of remaining layers will be incompatible once the layer is deleted</exception>
    public IFeedforwardNetworkLayer? RemoveLayer(int index) {
        if (index < 0)
            return null;
        
        if (index >= this.layers.Count)
            return null;

        int prev_index = index - 1;
        int next_index = index + 1;
        bool has_previous = prev_index >= 0;
        bool has_next = next_index < this.layers.Count;
        if (has_previous && has_next) {
            if (this.layers[prev_index].OutputShape != this.layers[next_index].InputShape) {
                throw new ArgumentException("Cannot remove layer as layer shapes will no longer be compatible");
            }
        }

        var layer = this.layers[index];
        this.layers.RemoveAt(index);
        return layer;
    }

    /// <summary>
    /// Replace the layer at the given index with a new layer
    /// </summary>
    /// <param name="index">index to replace layer at</param>
    /// <param name="layer">layer to replace with</param>
    /// <returns>The previous layer at that position</returns>
    /// <exception cref="ArgumentException">thrown when the new layer has an incompatible shape with the existing layer</exception>
    public IFeedforwardNetworkLayer? ReplaceLayer(int index, IFeedforwardNetworkLayer layer) {
        if (index < 0)
            return null;
        
        if (index >= this.layers.Count)
            return null;

        var current = this.layers[index];
        if (current.InputShape != layer.InputShape || current.OutputShape != layer.OutputShape) {
            throw new ArgumentException("Cannot replace layer as layer shapes will no longer be compatible");
        }
        this.layers[index] = layer;
        return current;
    }

    /// <summary>
    /// Add a layer to the end of the network
    /// </summary>
    /// <param name="layer">layer to add</param>
    /// <exception cref="ArgumentException">thrown when the layer has an incompatible shape with the prior layers</exception>
    public void AddLayer(IFeedforwardNetworkLayer layer) {
        if (this.layers.Count > 0) {
            var current_out_shape = this.OutputShape;
            if (!layer.DoesShapeMatchInputShape(current_out_shape)) {
                throw new ArgumentException("Layer input shape is incompatible with prior layers");
            }
        }
        this.layers.Add(layer);
    }

    /// <summary>
    /// Insert a new layer at the given position
    /// </summary>
    /// <param name="index">index to insert at</param>
    /// <param name="layer">layer to insert</param>
    /// <exception cref="ArgumentException">thrown when the layer has an incompatible shape with the prior and next layers</exception>
    public void InsertLayer(int index, IFeedforwardNetworkLayer layer) {
        if (index < 0)
            return;
        if (index >= this.layers.Count)
            return;

        var prev_index = index - 1;
        var current = layers[index];
        if (prev_index >= 0) {
            if (!layer.DoesShapeMatchInputShape(layers[prev_index].OutputShape)) {
                throw new ArgumentException("Layer input shape is incompatible with prior layers");
            }
        }
        if (!current.DoesShapeMatchInputShape(layer.OutputShape)) {
            throw new ArgumentException("Layer output shape is incompatible with next layers");
        }

        this.layers.Insert(index, layer);
    }

    /// <summary>
    /// Insert a layer before any layers that meet the given condition
    /// </summary>
    /// <param name="layer">Layer to insert</param>
    /// <param name="selector">Predicate function to indicate when a layer should be inserted</param>
    public void InsertLayerBefore(IFeedforwardNetworkLayer layer, Func<int, IFeedforwardNetworkLayer, bool> selector) {
        for (var i = 0; i < this.layers.Count; i++) {
            if (selector(i, this.layers[i])) {
                InsertLayer(i, layer);
                i++;
                return;
            }
        }
    }

    /// <summary>
    /// Insert a layer before any layers that meet the given condition
    /// </summary>
    /// <param name="layer">Layer to insert</param>
    /// <param name="selector">Predicate function to indicate when a layer should be inserted</param>
    public void InsertLayersBefore(Func<IFeedforwardNetworkLayer, IFeedforwardNetworkLayer> layer_generator, Func<int, IFeedforwardNetworkLayer, bool> selector) {
        for (var i = 0; i < this.layers.Count; i++) {
            if (selector(i, this.layers[i])) {
                InsertLayer(i, layer_generator(this.layers[i]));
                i++;
            }
        }
    }

    /// <summary>
    /// Insert a layer after any layers that meet the given condition
    /// </summary>
    /// <param name="layer">Layer to insert</param>
    /// <param name="selector">Predicate function to indicate when a layer should be inserted</param>
    public void InsertLayerAfter(IFeedforwardNetworkLayer layer, Func<int, IFeedforwardNetworkLayer, bool> selector) {
        for (var i = 0; i < this.layers.Count; i++) {
            if (selector(i, this.layers[i])) {
                InsertLayer(i + 1, layer);
                i++; // Skip the layer we just inserted
                return;
            }
        }
    }

    /// <summary>
    /// Insert a layer after any layers that meet the given condition
    /// </summary>
    /// <param name="layer">Layer to insert</param>
    /// <param name="selector">Predicate function to indicate when a layer should be inserted</param>
    public void InsertLayersAfter(Func<IFeedforwardNetworkLayer, IFeedforwardNetworkLayer> layer_generator, Func<int, IFeedforwardNetworkLayer, bool> selector) {
        for (var i = 0; i < this.layers.Count; i++) {
            if (selector(i, this.layers[i])) {
                InsertLayer(i + 1, layer_generator(this.layers[i]));
                i++; // Skip the layer we just inserted
            }
        }
    }

    public void Initialize(IInitializer initializer) {
        foreach (var layer in this.layers) {
            layer.Initialize(initializer);
        }
    }

    /// <summary>
    /// Number of trainable parameters in this network
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public int TrainableParameterCount() => this.layers.Select(layer => layer.TrainableParameterCount()).Sum();

    /// <summary>
    /// Number of un-trainable parameters in this network
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public int UnTrainableParameterCount() => this.layers.Select(layer => layer.UnTrainableParameterCount()).Sum();

    /// <summary>
    /// Total storage size for all parameters in this network
    /// </summary>
    /// <returns>data storage size</returns>
    public DataSize StorageSize() => DataSize.FromValues64(TrainableParameterCount()); // 64 since all parameters are doubles

    public void ValidateSizes() {
        var input_size = this.InputShape;
        int layer_index = 0;
        foreach (var layer in this.layers) { 
            if (!layer.DoesShapeMatchInputShape(input_size)) {
                throw new ArgumentException($"Layer {layer_index} expects an input shape of {layer.InputShape} but is receiving an input of shape {input_size} from the previous layer.");
            }
            input_size = layer.OutputShape;
            layer_index++;
        }
    }

    public BatchedFeatureSet<double> PredictSync(BatchedFeatureSet<double> values, Action<IFeedforwardNetworkLayer, BatchedFeatureSet<double>>? before_layer, Action<IFeedforwardNetworkLayer, BatchedFeatureSet<double>>? after_layer) {
        var ishape = this.InputShape;

        if (values.Channels != ishape.Channels) {
            throw new ArgumentException($"Invalid number of channels for input. Expected {ishape.Channels}, got {values.Channels}.");
        }
        if (values.Columns != ishape.Columns)
            throw new ArgumentException($"Invalid channel width. Expected {ishape.Columns}, got {values.Columns}.");
        if (values.Rows != ishape.Rows)
            throw new ArgumentException($"Invalid channel height. Expected {ishape.Rows}, got {values.Rows}.");
    
        BatchedFeatureSet<double> input = values;
        ishape = new Shape3D(input.Channels, input.Rows, input.Columns);
        var layer_index = 0;
        foreach (var layer in this.layers) {
            if (!layer.DoesShapeMatchInputShape(ishape))
                throw new ArithmeticException($"Input of shape {input.Channels}x{ishape.Rows}x{ishape.Columns} is incompatible with layer {layer_index} input's of shape {layer.InputShape}.");
            before_layer?.Invoke(layer, input);
            input = layer.EvaluateSync(input);
            after_layer?.Invoke(layer, input);
            ishape = new Shape3D(input.Channels, input.Rows, input.Columns);
            layer_index++;
        }
        return input;
    }

    public BatchedFeatureSet<double> PredictSync(BatchedFeatureSet<double> values) {
        return PredictSync(values, null, null);
    }

    public Vec<double> PredictSync(FeatureSet<double> values) {
        var output = PredictSync(new BatchedFeatureSet<double>(values));
        return Vec<double>.Wrap(output[0][0].FlattenRows().ToArray());
    }

    public Vec<double> PredictSync(Vec<double> input) {
        var ishape = this.InputShape;
        return PredictSync((FeatureSet<double>)(input.Shape(
            new Shape2D(ishape.Rows, ishape.Columns), 
                ishape.Channels
            ).ToArray())
        ); 
    }

    /// <summary>
    /// Network name, can be used to identify networks or their configuration
    /// </summary>
    public string? Name {get; set;}

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

    public void ToSvg(TextWriter s) {
        // Draw input layer

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
                case DenseLinearLayer connect:
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
        s.WriteLine($"<svg width='{img_width}{units}' height='{img_height}{units}' xmlns='http://www.w3.org/2000/svg'>");
        s.WriteLine("<defs>");
            s.WriteLine("<pattern id='grid' width='12' height='12' patternUnits='userSpaceOnUse'>");
                s.WriteLine("<rect x='0' y='0' width='12' height='12' fill='white'></rect>");
                s.WriteLine("<path d='M 12 0 L 0 0 0 12' fill='white' stroke='gray' stroke-width='1'/>");
            s.WriteLine("</pattern>");
        s.WriteLine("</defs>");

        // Draw "input" layer
        s.WriteLine($"<text x='{layer_width/2}' y='{16}' text-anchor='middle'>Input</text>");
        var input_count = this.InputShape.Channels;
        var matrix_offset = (layer_width - matrix_size) / 2;
        for (var i = 0; i < input_count; i++) {
            s.WriteLine($"<rect x='{matrix_offset}' y='{i * matrix_size + header_size}' width='{matrix_size}' height='{matrix_size}' fill='url(#grid)' stroke='black'></rect>");
        }
        var start_layer_midpoint_y = header_size + (input_count * matrix_size) / 2;

        // Draw all the rest of the layers
        last_output_matrices = input_count;
        var last_layer_midpoint_y = start_layer_midpoint_y;
        bool last_layer_was_fully_connected = false;
        for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
            s.WriteLine($"<g id='layer{layerIndex}'>");
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
                            s.WriteLine($"<rect x='{matrix_offset + layer_start_x + kernelIndex*kernel_offset}' y='{header_size + filterIndex * matrix_size - kernelIndex*kernel_offset}' width='{matrix_size}' height='{matrix_size}' fill='url(#grid)' stroke='black'></rect>");
                        }
                    }
                    layer_midpoint_y = header_size + (convo.FilterCount * matrix_size) / 2;
                    break;
                case PoolingLayer pool:
                    for (var i = 0; i < last_output_matrices; i++) {
                        s.WriteLine($"<rect x='{matrix_offset + layer_start_x}' y='{header_size + i * matrix_size}' width='{matrix_size}' height='{matrix_size}' fill='url(#grid)' stroke='black'></rect>");
                    }
                    layer_midpoint_y = last_layer_midpoint_y;
                    break;
                case ActivationLayer active:
                    {
                        s.WriteLine("<g id='synapses'>");
                        if (last_layer_was_fully_connected) {
                            for (var i = 0; i < active.InputShape.Count; i++) {
                                var neuron_offset = (matrix_size - 2*neuron_radius) / 2;
                                var center_x = matrix_offset + layer_start_x + matrix_size / 2;
                                var center_y = header_size + i * matrix_size + matrix_size / 2;
                                var in_center_x = (layer_buffer + layer_width) * layerIndex + matrix_offset + matrix_size - neuron_offset; // from the prev_layer
                                var in_center_y = header_size + i * matrix_size + matrix_size / 2;
                                s.WriteLine($"<line x1='{in_center_x}' y1='{in_center_y}' x2='{center_x - neuron_radius}' y2='{center_y}' stroke='gray'/>");
                            }
                        }
                        s.WriteLine("</g>");
                        for (var i = 0; i < active.InputShape.Count; i++) {
                            var center_x = matrix_offset + layer_start_x + matrix_size / 2;
                            var center_y = header_size + i * matrix_size + matrix_size / 2;
                            s.WriteLine($"<circle cx='{center_x}' cy='{center_y}' r='{neuron_radius}' fill='black' stroke='black'/>");
                            s.WriteLine($"<text x='{center_x}' y='{center_y}' text-anchor='middle' fill='white'>F(x)</text>");
                        }
                        layer_midpoint_y = last_layer_midpoint_y;
                    }
                    break;
                case DenseLinearLayer connect:
                    for (var i = 0; i < connect.OutputShape.Count; i++) {
                        var neuron_offset = (matrix_size - 2*neuron_radius) / 2;
                        var center_x = matrix_offset + layer_start_x + matrix_size / 2;
                        var center_y = header_size + i * matrix_size + matrix_size / 2;
                        s.WriteLine("<g id='synapses'>");
                        if (last_layer_was_fully_connected) {
                            for (var j = 0; j < connect.InputShape.Count; j++) {
                                var in_center_x = (layer_buffer + layer_width) * layerIndex + matrix_offset + matrix_size - neuron_offset; // from the prev_layer
                                var in_center_y = header_size + j * matrix_size + matrix_size / 2;
                                s.WriteLine($"<line x1='{in_center_x}' y1='{in_center_y}' x2='{center_x - neuron_radius}' y2='{center_y}' stroke='gray'/>");
                            }
                        }
                        s.WriteLine("</g>");
                        s.WriteLine($"<circle cx='{center_x}' cy='{center_y}' r='{neuron_radius}'/>");
                    }
                    layer_midpoint_y = header_size + (connect.OutputShape.Count * matrix_size) / 2;
                    break;
                case SoftmaxLayer softmax:
                    s.WriteLine($"<text x='{layer_width/2}' y='{layer_midpoint_y}' text-anchor='middle'>softmax(x)</text>");
                    break;
            }

            // Draw dotted arrow
            if (!last_layer_was_fully_connected) {
                s.WriteLine($"<line x1='{layer_start_buffer_x}' y1='{last_layer_midpoint_y}' x2='{layer_start_x}' y2='{layer_midpoint_y}' stroke-dasharray='4' stroke='gray'/>");
            }
            // Draw title
            s.WriteLine($"<text x='{layer_start_x + layer_width/2}' y='{layer_start_y + 16}' text-anchor='middle'>{layer.GetType().Name}</text>");


            switch (layer) {
                case ConvolutionLayer convo:
                    last_output_matrices = convo.FilterCount;
                    last_layer_was_fully_connected = false;
                    break;
                case PoolingLayer pool:
                    layer_midpoint_y = last_layer_midpoint_y;
                    last_layer_was_fully_connected = false;
                    break;
                case DenseLinearLayer connect:
                    last_output_matrices = 1;
                    last_layer_was_fully_connected = true;
                    break;
                case SoftmaxLayer:
                    last_output_matrices = 1;
                    last_layer_was_fully_connected = false;
                    break;
            }
            last_layer_midpoint_y = layer_midpoint_y;
            s.WriteLine("</g>");
        }

        s.Write("</svg>");
    }

    /// <summary>
    /// Convert this object to a Markdown representation
    /// </summary>
    /// <returns>Markdown serialized string</returns>
    public void ToMarkdown(TextWriter writer) {

        using (var md = new LayerMarkdownWriter(writer)) {
            for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
                var layer = this.GetLayer(layerIndex);
                if (!layer.Visit(md, layerIndex)) {
                    throw new ArgumentException($"Failed to print information for layer {layerIndex}.");
                }
            }
        }
        
    }

    /// <summary>
    /// Convert this object to an HTML representation
    /// </summary>
    /// <returns>HTML serialized string</returns>
    public void ToHtml(TextWriter writer) {

        using (var html = new LayerHtmlWriter(writer)) {
            for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
                var layer = this.GetLayer(layerIndex);
                if (!layer.Visit(html, layerIndex)) {
                    throw new ArgumentException($"Failed to print information for layer {layerIndex}.");
                }
            }
        }

    }
}