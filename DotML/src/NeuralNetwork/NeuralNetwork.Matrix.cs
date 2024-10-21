
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DotML.Network;

public class MatrixNeuralNetwork : ILayeredNeuralNetwork<MatrixNeuralNetwork.LayerReference> {

    public int InputCount {get; private set;}
    public int LayerCount {get; private set;}
    public int OutputCount {get; private set;}

    // Weights are 2D matrices stored "inbetween" layers
    /* IE:
        [w00 w10
         w01 w11
         w02 w12]
    */
    private double[][,] weights;
    // Biases are vectors stored with the layer
    /* IE:
        [b0
         b1
         b2]
    */
    private double[][] biases;
    private double[][] outputs;
    // Activation functions must be "uniform" per layer
    /* IE:
        y = h(w*x + b)
    */
    private ActivationFunction[] activations;

    public class NeuronReference : INeuron {
        private double[,] weights;
        private double[] biases;
        private ActivationFunction[] activations;
        private int layer_index;
        private int neuron_index;

        public NeuronReference(int layer, int neuron, double[,] weights, double[] biases, ActivationFunction[] activationFunctions) {
            this.layer_index = layer;
            this.neuron_index = neuron;
            this.weights = weights;
            this.biases = biases;
            this.activations = activationFunctions;
        }

        public double Bias { get => biases[neuron_index]; set => biases[neuron_index] = value; }

        public Span<double> Weights => MemoryMarshal.CreateSpan(ref weights[neuron_index, 0], weights.GetLength(1));

        public ActivationFunction? ActivationFunction {
            get => activations[layer_index];
            set => activations[layer_index] = value ?? Identity.Instance;
        }
    }

    public class LayerReference : ILayer {
        private double[] last_outputs;
        private NeuronReference[] neurons;
        private ActivationFunction[] activations;
        private int index;

        /// <summary>
        /// Number of inputs into the layer
        /// </summary>
        public int InputCount {get; init;}

        /// <summary>
        /// Number of outputs from the layer
        /// </summary>
        public int OutputCount {get; init;}

        public LayerReference(int index, int inputs, double[] last_outputs, NeuronReference[] neurons, ActivationFunction[] activations) {
            this.index = index;
            this.last_outputs = last_outputs;
            this.neurons = neurons;
            this.activations = activations;

            this.InputCount = inputs;
            this.OutputCount = last_outputs.Length;
        }

        public ActivationFunction ActivationFunction {
            get => activations[index];
            set => activations[index] = value;
        }

        public int NeuronCount => neurons.Length;

        public Vec<double> GetLastOutputs() => last_outputs;

        public INeuron GetNeuron(int index) {
            return neurons[index];
        }
    }

    private LayerReference[] layer_refs;

    public MatrixNeuralNetwork(int inputSize, params int[] layerSizes) {
        inputSize = Math.Max(1, inputSize);

        this.InputCount = inputSize;
        var layers = layerSizes.Length;
        this.LayerCount = layers;
        this.OutputCount = layerSizes[^1];

        // TODO make matrices and vectors of appropriate size

        // Make biases / activations
        this.biases = new double[layers][];
        this.outputs = new double[layers][];
        this.activations = new ActivationFunction[layers];
        for (var i = 0 ; i < layers; i++) {
            this.biases[i] = new double[layerSizes[i]];
            this.outputs[i] = new double[layerSizes[i]];
            this.activations[i] = Identity.Instance;
        }

        // Make weight matrices
        var inputs = inputSize;
        this.weights = new double[layers][,];
        for (var i = 0 ; i < layers; i++) {
            var outputs = layerSizes[i];
            this.weights[i] = new double[outputs,inputs]; // Not sure if right, tired today
            inputs = outputs;
        }

        // Make refs
        var neuron_refs = new NeuronReference[layers][]; 
        for (var i = 0; i < layers; i++) {
            var len = layerSizes[i];
            var refs = new NeuronReference[len];
            neuron_refs[i] = refs;

            for (var j = 0; j < len; j++) {
                refs[j] = new NeuronReference(i, j, weights[i], biases[i], activations);
            }
        }
        this.layer_refs = new LayerReference[layers];
        var layer_input = inputSize;
        for (var i = 0; i < layers; i++) {
            this.layer_refs[i] = new LayerReference(i, layer_input, outputs[i], neuron_refs[i], activations);
            layer_input = outputs[i].Length;
        }
    }

    public LayerReference GetLayer(int index) => layer_refs[index];

    public LayerReference GetFirstLayer() => layer_refs[0];

    public LayerReference GetOutputLayer() => layer_refs[^1];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double lerp(double min, double max, double sample) {
        return (Math.Max(min, max) * sample) + (Math.Min(min, max) * (1d - sample));
    }

    private static Random rng = new Random();

    public void RandomizeWeights(double min, double max) {
        for (var l = 0; l < weights.Length; l++) {
            var matrix = weights[l];
            for (var i = 0; i < matrix.GetLength(0); i++) {
                for (var j = 0; j < matrix.GetLength(1); j++) {
                    matrix[i,j] = lerp(min, max, rng.NextDouble());
                }
            }
        }
    }

    public void RandomizeBiases(double min, double max) {
        for (var l = 0; l < biases.Length; l++) {
            var vec = biases[l];
            for (var i = 0; i < vec.Length; i++) {
                vec[i] = lerp(min, max, rng.NextDouble());

            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void mul(double[,] mat, double[] vec, double[] res) {
        // matrix indicies are in row, col format
        var rows = mat.GetLength(0);
        var cols = mat.GetLength(1);

        for (var row = 0; row < rows; row++) {
            var sum = 0.0;
            for (var col = 0; col < cols; col++) {
                sum += mat[row, col] * vec[col];
            }
            res[row] = sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void add(double[] a, double[] b, double[] res) {
        var len = a.Length;
        for (var i = 0; i < len; i++) {
            res[i] = a[i] + b[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void operate(double[] a, ActivationFunction func, double[] res) {
        var len = a.Length;
        for (var i = 0; i < len; i++) {
            res[i] = func.Invoke(a[i]);
        } 
    }

    public Vec<double> PredictSync(Vec<double> input) {
        double[] in_vec = (double[])input;
        double[] out_vec;

        for (var i = 0; i < LayerCount; i++) {
            out_vec = outputs[i];

            mul(weights[i], in_vec, out_vec);
            add(out_vec, biases[i], out_vec);
            operate(out_vec, activations[i], out_vec);

            in_vec = out_vec;
        }
        return in_vec;
    }

    public Task<Vec<double>> PredictAsync(Vec<double> input) {
        return Task.Run(() => PredictSync(input));
    }
}