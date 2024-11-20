using System.Diagnostics.CodeAnalysis;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// A simple, fully connected, Feed Forward Neural Network based on layers.
/// </summary>
public class ClassicalFeedforwardNetwork : ILayeredNeuralNetwork<ILayerWithNeurons>, IJsonizable, IDiagrammable, ISafetensorable {
    private List<NeuronLayer> layers = new List<NeuronLayer>();

    /// <summary>
    /// The number of input values (size of input vector) allowed by this network.
    /// </summary>
    [JsonIgnore] public int InputCount => GetFirstLayer().InputCount;

    /// <summary>
    /// The number of output values (size of output vector) created by evaluating this network.
    /// </summary>
    [JsonIgnore] public int OutputCount => GetOutputLayer().OutputCount;

    /// <summary>
    /// Number of layers in this neural network.
    /// </summary>
    [JsonIgnore] public int LayerCount => layers.Count;

    /// <summary>
    /// Size of this neural network, number of neurons
    /// </summary>
    /// <returns>number of neurons in all layers</returns>
    [JsonIgnore] public int NeuronCount => layers.Select(x => x.NeuronCount).Sum();

    /// <summary>
    /// Create an empty Feed Forward Neural Network
    /// </summary>
    public ClassicalFeedforwardNetwork() {}

    /// <summary>
    /// Create a Feed Forward Neural Network from the give layers
    /// </summary>
    /// <param name="layers">network layers</param>
    public ClassicalFeedforwardNetwork(params NeuronLayer[] layers) {
        this.layers.AddRange(layers);
    }
    
    /// <summary>
    /// Create a Feed Forward Neural Network from the give layers
    /// </summary>
    /// <param name="layers">network layers</param>
    public ClassicalFeedforwardNetwork(IEnumerable<NeuronLayer> layers) {
        this.layers.AddRange(layers);
    }   

    /// <summary>
    /// Create a Feed Forward Neural Network from the shape
    /// </summary>
    /// <param name="inputSize">size of the input vector</param>
    /// <param name="layerSizes">size of each layer</param>
    public ClassicalFeedforwardNetwork(int inputSize, params int[] layerSizes) {
        for (var i = 0; i < layerSizes.Length; i++){
            this.layers.Add(new NeuronLayer(i == 0 ? inputSize : layerSizes[i-1], layerSizes[i]));
        }
    }

    /// <summary>
    /// Enumerate over all layers in this network
    /// </summary>
    /// <value>enumerable of layers</value>
    public IEnumerable<NeuronLayer> Layers {
        get {
            foreach (var layer in this.layers) {
                yield return layer;
            }
        }
        set {
            this.layers.Clear();
            this.layers.AddRange(value);
        }
    }

    public void Initialize(IInitializer initializer) {
        this.ForeachLayer(layer => {
            layer.ForeachNeuron((ref Neuron neuron) => {
                neuron.Bias = initializer.RandomBias(layer.NeuronCount);
                var weights = neuron.Weights;
                var weightc = weights.Length; 

                for (var i = 0; i < weightc; i++) {
                    weights[i] = initializer.RandomWeight(layer.InputCount + layer.OutputCount);
                }
            });
        });
    }

    /// <summary>
    /// Apply an action to every layer in the network
    /// </summary>
    /// <param name="action">action to apply to each layer</param>
    public void ForeachLayer(Action<NeuronLayer> action) {
        foreach (var layer in layers) {
            action(layer);
        }
    }

    /// <summary>
    /// Perform an action on each neuron in the network
    /// </summary>
    /// <param name="action">action to perform</param>
    public void ForeachNeuron(Action<ILayerWithNeurons, INeuron> action) {
        var layers = this.LayerCount;
        for (var layer_index = 0; layer_index < layers; layer_index++) {
            var layer = this.GetLayer(layer_index);
            var neurons = layer.NeuronCount;

            for (var neuron_index = 0; neuron_index < neurons; neuron_index++) {
                var neuron = layer.GetNeuron(neuron_index);
                action(layer, neuron);
            }
        }
    }

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public int TrainableParameterCount() => this.layers.Select(layer => layer.TrainableParameterCount()).Sum();

    /// <summary>
    /// Get a particular layer by index
    /// </summary>
    /// <param name="index">index to layer</param>
    /// <returns>layer</returns>
    public ILayerWithNeurons GetLayer(int index) {
        return this.layers[index];
    }

    /// <summary>
    /// Gets the first hidden layer in the network 
    /// </summary>
    /// <returns>layer</returns>
    public ILayerWithNeurons GetFirstLayer() {
        return this.layers[0];
    }

    /// <summary>
    /// Gets the last layer in the network where outputs are produced
    /// </summary>
    /// <returns>layer</returns>
    public ILayerWithNeurons GetOutputLayer() {
        return this.layers[this.layers.Count - 1];
    }

    /// <summary>
    /// Add a single layer to this network
    /// </summary>
    /// <param name="layer">layer to add</param>
    public void AddLayer(NeuronLayer layer) {
        layers.Add(layer);
    }

    /// <summary>
    /// Add many layers to this network
    /// </summary>
    /// <param name="layers">layers to add</param>
    public void AddLayers(params NeuronLayer[] layers) {
        this.layers.AddRange(layers);
    }

    /// <summary>
    /// Feed forward evaluation of this neural network
    /// </summary>
    /// <param name="input">input vector</param>
    /// <returns>outputs</returns>
    public Vec<double> PredictSync(Vec<double> input) {
        var inputs = input;
        foreach (var layer in layers) {
            var outputs = layer.EvaluateSync(inputs);
            inputs = outputs;
        }
        return inputs.Clone(); // Deep clone this since the outputs are just wrapped
    }
    
    /// <summary>
    /// Feed forward evaluation of this neural network using multiple Tasks
    /// </summary>
    /// <param name="input">input vector</param>
    /// <returns>outputs</returns>
    public async Task<Vec<double>> PredictAsync(Vec<double> input) {
        // TODO 
        // https://stackoverflow.com/questions/7612602/why-cant-i-use-the-await-operator-within-the-body-of-a-lock-statement
        // private SemaphoreSlim asyncLock = new SemaphoreSlim(1, 1);
        // Due to layers storing their outputs within themselves, we don't want to let someone else
        // start to modify the outputs while this is passing those same outputs to another layer
        // as this would create bad results. 
        // AKA a network in this config SHOULD only have 1 Evaluate going at a time.
        // Only enforcing this for Evaluate Async, So technically we can still break it by running 
        // an EvaluateAsync at the same time as an EvaluateSync 
        var inputs = input;
        foreach (var layer in layers) {
            var outputs =  await layer.EvaluateAsync(inputs);
            inputs = outputs;
        }
        return inputs.Clone(); // Deep clone this since the outputs are just wrapped
    }

    /// <summary>
    /// Output this network's configuration in the safetensor format
    /// </summary>
    /// <param name="writer">binary writer to write to</param>
    public void ToSafetensor(BinaryWriter writer) {
        // Header
        var max_layer_size = this.layers.Select(x => x.NeuronCount).Max();
        var weights_length = this.LayerCount * max_layer_size * max_layer_size;
        var biases_length = this.LayerCount * max_layer_size;

        var header = new {
            weights = new {
                dtype = "F64",
                shape = new int[]{this.LayerCount, max_layer_size, max_layer_size},
                data_offsets = new int[]{0, 0 + weights_length},
                __metadata__ = new {
                    use = "Weights for all synapses for each neuron",
                } 
            },
            biases = new {
                dtype = "F64",
                shape = new int[]{this.LayerCount, max_layer_size},
                data_offsets = new int[]{weights_length, weights_length + biases_length},
                __metadata__ = new {
                    use = "Biases for each neuron"
                } 
            },
        };
        var header_bytes = System.Text.Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        writer.Write(header_bytes.Length);
        writer.Write(header_bytes);

        // Weight tensor
        for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
            var layer = layers[layerIndex];

            for (var neuronIndex = 0; neuronIndex < max_layer_size; neuronIndex++) {
                if (neuronIndex >= layer.NeuronCount) {
                    writer.Write((byte)0);
                    continue;
                }

                INeuron neuron = layer.GetNeuron(neuronIndex);
                var weights = neuron.Weights;
                var len = weights.Length;

                for (var weightIndex = 0; weightIndex < len; weightIndex++) {
                    writer.Write(neuron.Weights[weightIndex]);
                }
            }
        }
        // Bias tensor
        for (var layerIndex = 0; layerIndex < this.LayerCount; layerIndex++) {
            var layer = layers[layerIndex];

            for (var neuronIndex = 0; neuronIndex < max_layer_size; neuronIndex++) {
                if (neuronIndex >= layer.NeuronCount) {
                    writer.Write((byte)0);
                    continue;
                }

                INeuron neuron = layer.GetNeuron(neuronIndex);
                writer.Write(neuron.Bias);
            }
        }
    }

    /// <summary>
    /// Convert this network to an SVG representation
    /// </summary>
    /// <returns>svg string</returns>
    public string ToSvg() {
        StringBuilder s = new StringBuilder();
        Random rng = new Random();

        var circle_radius = 24;
        var circle_diameter = circle_radius * 2;
        var circle_buffer = 24;
        var layer_width = circle_diameter + (2 * circle_buffer);
        var input_neuron_count = layers.Count > 0 ? layers[0].InputCount : 0;
        var layer_height = Math.Max(this.layers.Select(x => x.NeuronCount).Max(), input_neuron_count) * (circle_diameter + 2*circle_buffer);

        var gap_between_layers = 48;

        var neuron_block_width = layer_width;
        var neuron_block_height = layer_width;

        var img_width = layer_width * (this.LayerCount + 1) + Math.Max(0, gap_between_layers * (this.LayerCount));
        var img_height = layer_height;

        var units = "px";

        s.AppendLine($"<svg width='{img_width}{units}' height='{img_height}{units}' xmlns='http://www.w3.org/2000/svg'>");

        s.AppendLine("    <style>");
        s.AppendLine("        .Neuron  { stroke-width: 1px; stroke: black; fill: white; }");
        s.AppendLine("        .Synapse { stroke-width: 2px; stroke: black; }");
        s.AppendLine("        .Input   { fill: black; }");
        s.AppendLine("        .Bias    { fill: black; }");
        s.AppendLine("        .Weight  { fill: white; stroke-width: 1px; stroke: black; }");
        s.AppendLine("    </style>");

        // Synapses
        s.AppendLine("    <g id='Synapses'>");
        var synapse_from_layer_index = 0;
        var synapse_to_layer_index = 1;
        foreach (var layer in layers) {
            var gap = synapse_from_layer_index * gap_between_layers;
            var layer_start_x = gap + synapse_to_layer_index * layer_width;
            var layer_center_x = gap + (synapse_to_layer_index + 0.5f) * layer_width;
            var layer_end_x  = gap + (synapse_to_layer_index + 1) * layer_width;

            var neuron_index = 0;
            foreach (var neuron in layer.Neurons) {
                var neuron_center_x = gap_between_layers + layer_center_x - circle_radius;
                var neuron_center_y = neuron_index * neuron_block_height + circle_buffer + circle_radius;

                if (neuron.Weights is not null) {
                    var weight_index = 0;
                    foreach (var weight in neuron.Weights) {
                        var input_center_x = gap + (synapse_from_layer_index + 0.5f) * layer_width + circle_radius;
                        var input_center_y = weight_index * neuron_block_height + circle_buffer + circle_radius;
                        var synapse_label_loc_percent = rng.NextDouble();
                        var synapse_label_x = (1 - synapse_label_loc_percent) * input_center_x + synapse_label_loc_percent * neuron_center_x;
                        var synapse_label_y = (1 - synapse_label_loc_percent) * input_center_y + synapse_label_loc_percent * neuron_center_y;

                        s.Append("        ");
                        s.AppendLine($"<line id='Synapse {synapse_from_layer_index}[{weight_index}]->{synapse_to_layer_index}[{neuron_index}]' class='Synapse' x1='{input_center_x}{units}' y1='{input_center_y}{units}' x2='{neuron_center_x}{units}' y2='{neuron_center_y}{units}'/>");

                        var anchor = synapse_label_loc_percent switch {
                            < 0.25 => "start",
                            >= 0.25 and <= 0.75 => "middle",
                            > 0.75 => "end",
                            _ => "middle"
                        } ;

                        s.Append("        ");
                        s.AppendLine($"<text class='Weight' text-anchor='{anchor}' dominant-baseline=\"middle\" x='{synapse_label_x}{units}' y='{synapse_label_y}{units}'>{weight:F2}</text>");

                        weight_index++;
                    }
                }
                neuron_index++;
            }

            synapse_from_layer_index++;
            synapse_to_layer_index++;
        }
        s.AppendLine("    </g>");

        // Neurons
        s.AppendLine("    <g id='Neurons'>");
        if (this.LayerCount > 0) {
            // Input Layer
            var first_layer = this.layers[0];
            var last_layer = this.layers[^1];

            s.Append("        "); s.AppendLine($"<g id='Input Layer'>");
            for (var i = 0; i < input_neuron_count; i++) {
                var neuron_center_x = circle_buffer + circle_radius;
                var neuron_center_y = i * neuron_block_height + circle_buffer + circle_radius;
                s.Append("            ");
                s.AppendLine($"<circle id='Input {i + 1}' class='Neuron' r='{circle_radius}{units}' cx='{neuron_center_x}{units}' cy='{neuron_center_y}{units}'/>");
                s.Append("            ");
                s.AppendLine($"<text class='Input' text-anchor='middle' dominant-baseline=\"middle\" x='{neuron_center_x}{units}' y='{neuron_center_y}{units}'>IN {i+1}</text>");
            }
            s.Append("        "); s.AppendLine("</g>");

            // Hidden / Output Layers
            var layer_index = 1;
            foreach (var layer in this.layers) {
                var gap = (layer_index) * gap_between_layers;
                var layer_start_x = gap + layer_index * layer_width;
                var layer_center_x = gap + (layer_index + 0.5f) * layer_width;
                var layer_end_x  = gap + (layer_index + 1) * layer_width;

                s.Append("        "); if (layer == last_layer) { s.AppendLine($"<g id='Output Layer'>"); } else { s.AppendLine($"<g id='Hidden Layer {layer_index}'>"); }

                var neuron_index = 0;
                foreach (var neuron in layer.Neurons) {
                    var neuron_center_x = layer_center_x;
                    var neuron_center_y = neuron_index * neuron_block_height + circle_buffer + circle_radius;

                    s.Append("            ");
                    s.AppendLine($"<circle id='Neuron {neuron_index + 1}' class='Neuron' r='{circle_radius}{units}' cx='{neuron_center_x}{units}' cy='{neuron_center_y}{units}'>");
                    s.Append("                ");
                    s.AppendLine($"<desc id='Bias'>{neuron.Bias}</desc>");
                    s.Append("                ");
                    s.AppendLine($"<desc id='Activation Function'>{neuron.ActivationFunction?.GetType()?.Name}</desc>");
                    s.Append("            ");
                    s.AppendLine($"</circle>");
                    s.Append("            ");
                    s.AppendLine($"<text class='Bias' text-anchor='middle' dominant-baseline=\"middle\" x='{neuron_center_x}{units}' y='{neuron_center_y}{units}'>{neuron.Bias:F2}</text>");

                    neuron_index++;
                }
                s.Append("        "); s.AppendLine("</g>");

                layer_index++;
            }
        }
        s.AppendLine("    </g>");

        s.Append("</svg>");
        return s.ToString();
    }

    /// <summary>
    /// Convert this network to a JSON representation
    /// </summary>
    /// <returns>JSON serialized Neural Network string</returns>
    public string ToJson() {
        return JsonSerializer.Serialize(this);
    }
    /// <summary>
    /// Create a network from a JSON representation
    /// </summary>
    /// <param name="str">JSON text</param>
    /// <returns>Deserialized Neural Network</returns>
    public static ClassicalFeedforwardNetwork FromJson(string str) {
        var network = JsonSerializer.Deserialize<ClassicalFeedforwardNetwork>(str);
        if (network is null)
            return new ClassicalFeedforwardNetwork();
        return network;
    }
    /// <summary>
    /// Create a network from a JSON representation
    /// </summary>
    /// <param name="str">JSON text</param>
    /// <param name="network">Deserialized Neural Network</param>
    /// <returns>true if the network was successfully deserialized</returns>
    public static bool TryFromJson(string str, [NotNullWhen(true)]out ClassicalFeedforwardNetwork? network) {
        try {
            network = JsonSerializer.Deserialize<ClassicalFeedforwardNetwork>(str);
            if (network is null)
                throw new ArgumentNullException();
            return true;
        } catch {
            network = null;
            return false;
        }
    }
}