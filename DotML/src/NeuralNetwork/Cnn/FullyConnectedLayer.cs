using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Fully connected layer which flattens it's inputs before processing and returns a column vector from the output neurons
/// </summary>
public class FullyConnectedLayer : ConvolutionalFeedforwardNetworkLayer, ILayerWithNeurons {

    private int inputs;
    private int outputs;
    private int neuronc;
    public int NeuronCount=> neuronc;
 
    public Matrix<double> Weights {
        get => Matrix<double>.Wrap(weight_values);
        set {
            weight_values = (double[,])value;
        }
    }
    private double[,] weight_values;
    public Vec<double> Biases {
        get => Vec<double>.Wrap(bias_values);
        set {
            bias_values = (double[])value;
        }
    }
    private double[] bias_values;

    public ActivationFunction ActivationFunction {get; set;}
 
    /// <summary>
    /// Neuron interface to access individual neuron properties from matrix and vector data
    /// </summary>
    public class NeuronRef : INeuron {
        FullyConnectedLayer parent; int index;
        internal NeuronRef(FullyConnectedLayer parent, int index) {
            this.index = index;
            this.parent = parent;
        }

        /// <summary>
        /// Neuron bias
        /// </summary>
        public double Bias {
            get => parent.bias_values[index];
            set => parent.bias_values[index] = value;
        }

        /// <summary>
        /// Neuron synapse weights
        /// </summary>
        // TODO broken. Weights are rows per neuron, not columns
        public Span<double> Weights {
            get => MemoryMarshal.CreateSpan(ref parent.weight_values[index, 0], parent.weight_values.GetLength(1));
            set {
                for (var i = 0; i < parent.weight_values.GetLength(0); i++) {
                    if (index < value.Length)
                        parent.weight_values[i, index] = value[i];
                    else    
                        parent.weight_values[i, index] = 0.0;
                }
            }
        }

        /// <summary>
        /// Neuron activation function
        /// </summary>
        public ActivationFunction? ActivationFunction {
            get => parent.ActivationFunction;
        }
    }

    private NeuronRef[] neurons;

    public FullyConnectedLayer(int input_size, int neurons, ActivationFunction? activation = null) {
        this.inputs = input_size;
        this.outputs = neurons;
        this.neuronc = neurons;
        this.Weights = new Matrix<double>(neurons, input_size); // #rows = output count = neurons; #columns = input count
        this.weight_values = (double[,])Weights;
        this.Biases = new Vec<double>(neurons);
        this.bias_values = (double[])Biases;
        this.ActivationFunction = activation ?? Identity.Instance;

        this.InputShape = new Shape3D(1, input_size, 1);
        this.OutputShape = new Shape3D(1, neurons, 1);

        this.neurons = Enumerable.Range(0, neurons).Select(x => new NeuronRef(this, x)).ToArray();
    }

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public override int TrainableParameterCount() => Weights.Size + Biases.Dimensionality;

    /// <summary>
    /// Get a given neuron from this layer
    /// </summary>
    /// <param name="index">neuron index</param>
    /// <returns>neuron reference</returns>
    public INeuron GetNeuron(int index) => this.neurons[index];

    public override void Initialize(IInitializer initializer) {
        for (var b = 0; b < bias_values.Length; b++) {
            bias_values[b] = initializer.RandomBias(bias_values.Length);
        }

        var parameters = this.InputShape.Count + this.OutputShape.Count;
        for (var i = 0; i < weight_values.GetLength(0); i++) {
            for (var j = 0; j < weight_values.GetLength(1); j++) {
                weight_values[i, j] = initializer.RandomWeight(parameters);
            }
        }
    }

    public override void Visit(IConvolutionalLayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddMatVecInplace(Matrix<double> target, Matrix<double> a, Vec<double> b) {
        var result = (double[,])target;
        var rows = a.Rows;
        if (rows != target.Rows) {
            throw new ArithmeticException("Incompatible dimensions for storing results of matrix/vector addition");
        }
        for (var row = 0; row < rows; row++) {
            result[row, 0] = a[row, 0] + b[row];
        }
    }

    public override bool DoesShapeMatchInputShape(Matrix<double>[] channels) {
        return channels.Select(c => c.Size).Sum() == inputs;
    }

    public override Matrix<double>[] EvaluateSync(Matrix<double>[] inputs) {
        // input is a 2D matrix processed from prior layers like a pooling layer
        var x = inputs.Length == 1 && inputs[0].IsColumn ? inputs[0] : Matrix<double>.Column(inputs.SelectMany(x => x.FlattenRows()).ToArray());
        //var x = Matrix<double>.Column(inputs.SelectMany(x => x.FlattenRows()).ToArray()); 
        var mul  = Weights * x; 
        AddMatVecInplace(mul, mul, bias_values);
        Matrix<double>.TransformInplace(mul, mul, this.ActivationFunction.Invoke);
        return [ mul ]; 
        //var biased = mul + Matrix<double>.Column(bias_values); 
        //var activated = this.ActivationFunction.Invoke(biased);
        //return [ activated ];
    }

}