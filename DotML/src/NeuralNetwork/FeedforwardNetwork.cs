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
    public void InsertLayersBeforeAll(Func<IFeedforwardNetworkLayer, IFeedforwardNetworkLayer> layer_generator, Func<int, IFeedforwardNetworkLayer, bool> selector) {
        for (var i = 0; i < this.layers.Count; i++) {
            if (selector(i, this.layers[i])) {
                InsertLayer(i, layer_generator(this.layers[i]));
                i++;
            }
        }
    }

    /// <summary>
    /// Insert a layer before the first layer that meet the given condition
    /// </summary>
    /// <param name="layer">Layer to insert</param>
    /// <param name="selector">Predicate function to indicate when a layer should be inserted</param>
    public void InsertLayerBefore(Func<IFeedforwardNetworkLayer, IFeedforwardNetworkLayer> layer_generator, Func<int, IFeedforwardNetworkLayer, bool> selector) {
        for (var i = 0; i < this.layers.Count; i++) {
            if (selector(i, this.layers[i])) {
                InsertLayer(i, layer_generator(this.layers[i]));
                i++;
                return;
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
    public void InsertLayersAfterAll(Func<IFeedforwardNetworkLayer, IFeedforwardNetworkLayer> layer_generator, Func<int, IFeedforwardNetworkLayer, bool> selector) {
        for (var i = 0; i < this.layers.Count; i++) {
            if (selector(i, this.layers[i])) {
                InsertLayer(i + 1, layer_generator(this.layers[i]));
                i++; // Skip the layer we just inserted
            }
        }
    }

    /// <summary>
    /// Insert a layer after the first layer that meet the given condition
    /// </summary>
    /// <param name="layer">Layer to insert</param>
    /// <param name="selector">Predicate function to indicate when a layer should be inserted</param>
    public void InsertLayerAfter(Func<IFeedforwardNetworkLayer, IFeedforwardNetworkLayer> layer_generator, Func<int, IFeedforwardNetworkLayer, bool> selector) {
        for (var i = 0; i < this.layers.Count; i++) {
            if (selector(i, this.layers[i])) {
                InsertLayer(i + 1, layer_generator(this.layers[i]));
                i++; // Skip the layer we just inserted
                return;
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
        if (input_size.Channels < 0 || input_size.Rows < 0 || input_size.Columns < 0) {
            throw new ArgumentException("Input shape is invalid");
        }
        int layer_index = 0;
        foreach (var layer in this.layers) { 
            if (!layer.DoesShapeMatchInputShape(input_size)) {
                throw new ArgumentException($"Layer {layer_index} expects an input shape of {layer.InputShape} but is receiving an input of shape {input_size} from the previous layer.");
            }
            input_size = layer.OutputShape;
            if (input_size.Channels < 0 || input_size.Rows < 0 || input_size.Columns < 0) {
                throw new ArgumentException($"Output shape for layer {layer_index} is invalid");
            }
            layer_index++;
        }
    }

    public bool HasValidSizes() {
        try {
            ValidateSizes();
            return true;
        } catch {
            return false;
        }
    }

    public BatchedFeatureSet<float> PredictSync(BatchedFeatureSet<float> values, Action<IFeedforwardNetworkLayer, BatchedFeatureSet<float>>? before_layer, Action<IFeedforwardNetworkLayer, BatchedFeatureSet<float>>? after_layer) {
        var ishape = this.InputShape;

        if (values.Channels != ishape.Channels) {
            throw new ArgumentException($"Invalid number of channels for input. Expected {ishape.Channels}, got {values.Channels}.");
        }
        if (values.Columns != ishape.Columns)
            throw new ArgumentException($"Invalid channel width. Expected {ishape.Columns}, got {values.Columns}.");
        if (values.Rows != ishape.Rows)
            throw new ArgumentException($"Invalid channel height. Expected {ishape.Rows}, got {values.Rows}.");
    
        BatchedFeatureSet<float> input = values;
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

    public BatchedFeatureSet<float> PredictSync(BatchedFeatureSet<float> values) {
        return PredictSync(values, null, null);
    }

    public Vec<float> PredictSync(FeatureSet<float> values) {
        var output = PredictSync(new BatchedFeatureSet<float>(values));
        return Vec<float>.Wrap(output.FlattenElements().ToArray());
    }

    public Vec<float> PredictSync(Vec<float> input) {
        var ishape = this.InputShape;
        return PredictSync((FeatureSet<float>)(input.Shape(
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
            layer.Visit(reader, layerIndex);
        }
    }

    /// <summary>
    /// Output this network's configuration in the safetensor format
    /// </summary>
    public Safetensors ToSafetensor() {
        var writer = new LayerSafetensorWriter();

        for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
            var layer = this.GetLayer(layerIndex);
            layer.Visit(writer, layerIndex);
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
        var writer = new SvgWriter();
        writer.WriteTo(this, s);
    }

    /// <summary>
    /// Convert this object to a Markdown representation
    /// </summary>
    /// <returns>Markdown serialized string</returns>
    public void ToMarkdown(TextWriter writer) {

        using (var md = new LayerMarkdownWriter(writer)) {
            for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
                var layer = this.GetLayer(layerIndex);
                layer.Visit(md, layerIndex);
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
                layer.Visit(html, layerIndex);
            }
        }

    }
}