using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Fully connected layer which flattens it's inputs before processing and returns a column vector from the output neurons
/// </summary>
public class FullyConnectedLayer : FeedforwardNetworkLayer, ILayerWithNeurons {

    private int inputs;
    private int outputs;
    private int neuronc;
    public int NeuronCount=> neuronc;
    
    [JsonIgnore]
    public Matrix<double> Weights {
        get; set;
    }

    [JsonIgnore]
    public Vec<double> Biases {
        get => Vec<double>.Wrap(bias_values);
        set {
            bias_values = (double[])value;
        }
    }
    private double[] bias_values;
 
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
            get => parent.Weights.ExtractRowSpan(index);
            set {
                var span = parent.Weights.ExtractRowSpan(index);
                value.CopyTo(span);
            }
        }

        /// <summary>
        /// Neuron activation function
        /// </summary>
        public ActivationFunction? ActivationFunction {
            get => Identity.Instance;
        }
    }

    private NeuronRef[] neurons;

    public FullyConnectedLayer(int input_size, int neurons) {
        this.inputs = input_size;
        this.outputs = neurons;
        this.neuronc = neurons;
        this.Weights = new Matrix<double>(neurons, input_size); // #rows = output count = neurons; #columns = input count
        this.Biases = new Vec<double>(neurons);
        this.bias_values = (double[])Biases;

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
        var parameters = this.TrainableParameterCount();

        for (var b = 0; b < bias_values.Length; b++) {
            bias_values[b] = initializer.RandomBias(this.InputShape.Count, this.OutputShape.Count, parameters);
        }

        var weights = this.Weights;
        for (var i = 0; i < weights.Rows; i++) {
            for (var j = 0; j < weights.Columns; j++) {
                weights[i, j] = initializer.RandomWeight(this.InputShape.Count, this.OutputShape.Count, parameters);
            }
        }
    }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(ILayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddMatVecInplace(Matrix<double> target, Matrix<double> a, Vec<double> b) {
        var result = target;
        var rows = a.Rows;
        if (rows != target.Rows) {
            throw new ArithmeticException("Incompatible dimensions for storing results of matrix/vector addition");
        }
        for (var i = 0; i < rows; i++) {
            result[i, 0] = a[i, 0] + b[i];
        }
    }

    // On Average, this is slightly faster than the old method but does the same thing (if I didn't copy-paste it wrong) will have to test this haha
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Matrix<double> MultiplyMatrixVectorAndAddVector(Matrix<double> a, Matrix<double> b, Vec<double> c) {
        // Remember that B is a vector
        //if (a.Columns != b.Rows || b.Columns != 1 || a.Rows != c.Dimensionality)
            //throw new ArithmeticException($"Incompatible dimensions for matrix multiplication {a.Rows}x{a.Columns} · {b.Rows}x{b.Columns} + {c.Dimensionality}x1");

        int rows = a.Rows;
        int cols = b.Columns; // aka 1
        int innerDim = a.Columns;

        var result = new Matrix<double>(rows, cols);
        for (var i = 0; i < rows; i++) {
            double sum = 0.0;     
            for (int j = 0; j < innerDim; j++) {
                sum += a[i, j] * b[j, 0];
            }
            result[i, 0] = sum + c[i];
        }
        return result;
    }

    public override bool DoesShapeMatchInputShape(Shape3D shape) {
        return shape.Count == inputs;
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> inputs) {
        // input is a 2D matrix processed from prior layers like a pooling layer
        var x = inputs.Channels == 1 && inputs[0].IsColumnMatrix ? inputs[0] : Matrix<double>.Column(inputs.SelectMany(x => x.FlattenRows()).ToArray());
        //var x = Matrix<double>.Column(inputs.SelectMany(x => x.FlattenRows()).ToArray()); 
        //var mul  = Weights * x; 
        //AddMatVecInplace(mul, mul, bias_values);
        //return new FeatureSet<double>(mul); 
        return new FeatureSet<double>(MultiplyMatrixVectorAndAddVector(Weights, x, bias_values));
        //var biased = mul + Matrix<double>.Column(bias_values); 
        //var activated = this.ActivationFunction.Invoke(biased);
        //return [ activated ];
    }

}