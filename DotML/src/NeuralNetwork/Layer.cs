using System.Text.Json.Serialization;

namespace DotML.Network;

/// <summary>
/// Basic layer of neurons
/// </summary>
public class Layer : ILayer {
    private Neuron[] neurons;
    private LayerNeuronReference[] neuron_refs;
    private double[] outputs; // Outputs will be cached

    /// <summary>
    /// Size of the layer, number of neurons
    /// </summary>
    [JsonIgnore] public int NeuronCount => neurons.Length;

    /// <summary>
    /// Neurons within this layer
    /// </summary>
    /// <value>enumerable of neurons</value> 
    public IEnumerable<Neuron> Neurons {
        get {
            for (var i = 0; i < NeuronCount; i++) {
                yield return neurons[i];
            }
        } set {
            this.neurons = value.ToArray();
            this.outputs = new double[this.neurons.Length];
            this.neuron_refs = new LayerNeuronReference[this.neurons.Length];
            for (int i = 0; i < this.neurons.Length; i++) {
                neuron_refs[i] = new LayerNeuronReference(neurons, i);
            }
        }
    }

    /// <summary>
    /// All neuron biases within this layer
    /// </summary>
    /// <value>enumerable of biases</value> 
    [JsonIgnore] public IEnumerable<double> Biases {
        get {
            for (var i = 0; i < NeuronCount; i++) {
                yield return neurons[i].Bias;
            }
        } set {
            var enumerator = value.GetEnumerator();
            for (var i = 0; i < NeuronCount; i++) {
                if (!enumerator.MoveNext())
                    continue;
                neurons[i].Bias = enumerator.Current;
            }
        }
    }

    /// <summary>
    /// All neuron input weights within this layer
    /// </summary>
    /// <value>enumerable of weights</value> 
    [JsonIgnore] public IEnumerable<double[]> Weights {
        get {
            for (var i = 0; i < NeuronCount; i++) {
                yield return neurons[i].Weights.ToArray();
            }
        } set {
            var enumerator = value.GetEnumerator();
            for (var i = 0; i < NeuronCount; i++) {
                if (!enumerator.MoveNext())
                    continue;
                neurons[i].Weights = enumerator.Current;
            }
        }
    }

    /// <summary>
    /// All neuron activation functions within this layer
    /// </summary>
    /// <value>enumerable of activation functions</value> 
    [JsonIgnore] public IEnumerable<ActivationFunction> ActivationFunctions {
        get {
            for (var i = 0; i < NeuronCount; i++) {
                yield return neurons[i].ActivationFunction ?? Identity.Instance;
            }
        } set {
            var enumerator = value.GetEnumerator();
            for (var i = 0; i < NeuronCount; i++) {
                if (!enumerator.MoveNext())
                    continue;
                neurons[i].ActivationFunction = enumerator.Current;
            }
        }
    }

    /// <summary>
    /// Size of the input to this layer
    /// </summary>
    [JsonIgnore] public int InputCount => Neurons.Select(n => n.Weights.Length).Max();

    /// <summary>
    /// Number of outputs from the layer
    /// </summary>
    [JsonIgnore] public int OutputCount => neurons.Length;

    /// <summary>
    /// Deserialization constructor
    /// </summary>
    public Layer() {
        neurons = new Neuron[0];
        outputs = new double[0];
        neuron_refs = new LayerNeuronReference[0];
    }

    /// <summary>
    /// Create a new layer of the given size
    /// </summary>
    /// <param name="size">layer size, minimum 1</param>
    public Layer(int inputs, int size) {
        size = Math.Max(1, size);
        neurons = new Neuron[size];
        outputs = new double[size];
        neuron_refs = new LayerNeuronReference[size];

        for (int i = 0; i < size; i++) {
            neurons[i].Weights = new double[inputs];
            neuron_refs[i] = new LayerNeuronReference(neurons, i);
        }
    }

    /// <summary>
    /// The results of the last evaluation, for debugging
    /// </summary>
    public Vec<double> GetLastOutputs() => outputs;

    private static readonly Random rng = new Random();

    /// <summary>
    /// A heap based wrapper referring to a specific neuron in this layer
    /// </summary>
    public class LayerNeuronReference : INeuron {
        private Neuron[] array;
        private int index;

        internal LayerNeuronReference(Neuron[] array, int index) {
            this.array = array;
            this.index = index;
        }

        public double Bias { 
            get => array[index].Bias;
            set => array[index].Bias = value;
        }
        public Span<double> Weights { 
            get => array[index].Weights;
            set => array[index].Weights = value.ToArray();
        }
        public ActivationFunction? ActivationFunction { 
            get => array[index].ActivationFunction;
            set => array[index].ActivationFunction = value;
        }

        public double Evaluate(Vec<double> inputs) {
            return array[index].Evaluate(inputs);
        }
    }

    /// <summary>
    /// Get a reference to a specific neuron
    /// </summary>
    /// <param name="index">neuron index</param>
    /// <returns>neuron reference</returns>
    public INeuron GetNeuron(int index) {
        //if (index < 0 || index >= neurons.Length)
            //throw new ArgumentOutOfRangeException(nameof(index));

        return this.neuron_refs[index];
    }

    /// <summary>
    /// Apply an action to every neuron in the layer
    /// </summary>
    /// <param name="action">action to apply to each neuron</param>
    public void ForeachNeuron(NeuronAction action) {
        for (var i = 0; i < this.NeuronCount; i++) {
            ref Neuron neuron = ref neurons[i];
            action(ref neuron);
        }
    }

    /// <summary>
    /// Evaluate all neurons in this layer with the given inputs
    /// </summary>
    /// <param name="input">input vector</param>
    /// <returns>output vector</returns>
    public Vec<double> EvaluateSync(Vec<double> input) {
        var count = neurons.Length;
        if (outputs is null || outputs.Length != count)
            outputs = new double[count];

        Array.Fill(outputs, 0.0);

        // Evaluate neuron outputs in series
        //Parallel.For(0, count, (index, _) => {
            //outputs[index] = neurons[index].Evaluate(input);
        //});
        for (var i = 0; i < count; i++) {
            outputs[i] = neurons[i].Evaluate(input);
        }
        return new Vec<double>(outputs);
    }
    /// <summary>
    /// Evaluate all neurons in this layer with the given inputs asynchronously
    /// </summary>
    /// <param name="input">input vector</param>
    /// <returns>output vector</returns>
    public async Task<Vec<double>> EvaluateAsync(Vec<double> input) {
        return await Task.Run(() => {
            var count = NeuronCount;
            if (outputs is null || outputs.Length != count)
                outputs = new double[count];

            Array.Fill(outputs, 0.0);

            // Evaluate neuron outputs in parallel
            for (var i = 0; i < count; i++) {
                outputs[i] = neurons[i].Evaluate(input);
            }
            return new Vec<double>(outputs);
        });
    }
}