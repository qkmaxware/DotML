using System.Runtime.CompilerServices;

namespace DotML.Network.Training;

/// <summary>
/// Optimizer to adjust the learning rate during training.
/// </summary>
public interface ILearningRateOptimizer {
    /// <summary>
    /// Initialize the optimizer to work for a give network/network shape
    /// </summary>
    /// <param name="network">network and network shape</param>
    public void Initialize(ILayeredNeuralNetwork<ILayerWithNeurons> network);
    /// <summary>
    /// Return an adjusted learning rate based on the gradient of the weight at the given layer, neuron, synapse index
    /// </summary>
    /// <param name="baseLearningRate">Base learning rate</param>
    /// <param name="gradient">weight gradient</param>
    /// <param name="layer">layer index</param>
    /// <param name="neuron">neuron index</param>
    /// <param name="weight">synapse index</param>
    /// <returns>adjusted learning rate</returns>
    public double UpdateLearningRate(double baseLearningRate, double gradient, int layer, int neuron, int weight);
    /// <summary>
    /// Return an adjusted learning rate based on the gradient of the bias at the given layer, neuron index
    /// </summary>
     // <param name="baseLearningRate">Base learning rate</param>
    /// <param name="gradient">weight gradient</param>
    /// <param name="layer">layer index</param>
    /// <param name="neuron">neuron index</param>
    /// <returns>adjusted learning rate</returns>
    public double UpdateLearningRate(double baseLearningRate, double gradient, int layer, int neuron);
}

/// <summary>
/// Static rate optimizer. Keeps the learning rate static across the entire training session.
/// </summary>
public class ConstantRate : ILearningRateOptimizer {
    public void Initialize(ILayeredNeuralNetwork<ILayerWithNeurons> network) { /* No need to do anything */ }

    public double UpdateLearningRate(double baseLearningRate, double gradient, int layer, int neuron, int weight) {
        return baseLearningRate;
    }
    public double UpdateLearningRate(double baseLearningRate, double gradient, int layer, int neuron) {
        return baseLearningRate;
    }
}

/// <summary>
/// Learning rate optimizer based on the RMSProp technique. Learning rate gets smaller as the gradient gets smaller.
/// </summary>
public class RMSPropOptimizer : ILearningRateOptimizer {
    /// <summary>
    /// Decay rate for the learning rate
    /// </summary>
    public double DecayRate {get; init;}

    private double[][][]? weights_cache;
    private double[][]? bias_cache;

    const double epsilon = 1e-8;

    /// <summary>
    /// Learning rate optimizer based on the RMSProp technique. Learning rate gets smaller as the gradient gets smaller.
    /// </summary>
    /// <param name="decayRate">Decay rate for the learning rate</param>
    public RMSPropOptimizer(double decayRate = 0.9) {
        this.DecayRate = Math.Abs(decayRate); // Can't be -
    }

    public void Initialize(ILayeredNeuralNetwork<ILayerWithNeurons> network) {
        weights_cache = new double[network.LayerCount][][];
        for (var layer = 0; layer < weights_cache.Length; layer++) {
            var layerObj = network.GetLayer(layer);
            var lweights = new double[layerObj.NeuronCount][];
            weights_cache[layer] = lweights;

            for (var neuron = 0; neuron < lweights.Length; neuron++) {
                var nweights = new double[layerObj.GetNeuron(neuron).Weights.Length];
                lweights[neuron] = nweights;
                for (var w = 0; w < nweights.Length; w++) {
                    nweights[w] = 0d;
                }
            }
        }

        bias_cache = new double[network.LayerCount][]; 
        for (var layer = 0; layer < weights_cache.Length; layer++) {
            var layerObj = network.GetLayer(layer);
            var lweights = new double[layerObj.NeuronCount];
            bias_cache[layer] = lweights;
            for (var w = 0; w < lweights.Length; w++) {
                lweights[w] = 0d;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double getCached(int layer, int neuron, int weight) {
        if (weights_cache is null)
            return 0d;
        if (layer < 0 || layer >= weights_cache.Length)
            return 0d;
        var l = weights_cache[layer];

        if (neuron < 0 || neuron >= l.Length)
            return 0d;
        var n = l[neuron];

        if (weight < 0 || weight >= n.Length)
            return 0d;
        return n[weight];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double getCached(int layer, int neuron) {
        if (bias_cache is null)
            return 0d;
        if (layer < 0 || layer >= bias_cache.Length)
            return 0d;
        var l = bias_cache[layer];

        if (neuron < 0 || neuron >= l.Length)
            return 0d;
        var n = l[neuron];
        return n;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void setCached(int layer, int neuron, int weight, double value) {
        if (weights_cache is null)
            return;
        if (layer < 0 || layer >= weights_cache.Length)
            return;
        var l = weights_cache[layer];

        if (neuron < 0 || neuron >= l.Length)
            return;
        var n = l[neuron];

        if (weight < 0 || weight >= n.Length)
            return;
        n[weight] = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void setCached(int layer, int neuron, double value) {
        if (bias_cache is null)
            return;
        if (layer < 0 || layer >= bias_cache.Length)
            return;
        var l = bias_cache[layer];

        if (neuron < 0 || neuron >= l.Length)
            return;
        l[neuron] = value;
    }

    /*
        // Update cache with the squared gradient
        cache[index] = decayRate * cache[index] + (1 - decayRate) * gradient * gradient;

        // Compute the adjusted learning rate
        return learningRate / (Math.Sqrt(cache[index]) + epsilon);
    */

    public double UpdateLearningRate(double baseLearningRate, double gradient, int layer, int neuron, int weight) {
        var cached = DecayRate * getCached(layer, neuron, weight) + (1 - DecayRate) * gradient * gradient;
        setCached(layer, neuron, weight, cached);

        return baseLearningRate / (Math.Sqrt(cached) + epsilon);
    }

    public double UpdateLearningRate(double baseLearningRate, double gradient, int layer, int neuron) {
        var cached = DecayRate * getCached(layer, neuron) + (1 - DecayRate) * gradient * gradient;
        setCached(layer, neuron, cached);

        return baseLearningRate / (Math.Sqrt(cached) + epsilon);
    }
}

/// <summary>
/// Learning rate optimizer based on the ADAM technique. Learning rate gets smaller as the gradient gets smaller.
/// </summary>
public class AdamOptimizer : ILearningRateOptimizer {
    /// <summary>
    /// First ADAM hyperparameter
    /// </summary>
    public double Beta1 {get; init;}

    /// <summary>
    /// Second ADAM hyperparameter
    /// </summary>
    public double Beta2 {get; init;}

    private int timestep = 0;

    struct Moment {
        public double First;
        public double Second;
    }
    private Moment[][][]? m_weights;
    private Moment[][]? m_bias;

    public AdamOptimizer(double beta1 = 0.9, double beta2 = 0.999) {
        this.Beta1 = Math.Abs(beta1);
        this.Beta2 = Math.Abs(beta2);
    }

    public void Initialize(ILayeredNeuralNetwork<ILayerWithNeurons> network) {
        timestep = 0;
        
        m_weights = new Moment[network.LayerCount][][];
        for (var layer = 0; layer < m_weights.Length; layer++) {
            var layerObj = network.GetLayer(layer);
            var lweights = new Moment[layerObj.NeuronCount][];
            m_weights[layer] = lweights;

            for (var neuron = 0; neuron < lweights.Length; neuron++) {
                var nweights = new Moment[layerObj.GetNeuron(neuron).Weights.Length];
                lweights[neuron] = nweights;
                for (var w = 0; w < nweights.Length; w++) {
                    nweights[w] = new Moment{ First = 0d, Second = 0d };
                }
            }
        }

        m_bias = new Moment[network.LayerCount][]; 
        for (var layer = 0; layer < m_bias.Length; layer++) {
            var layerObj = network.GetLayer(layer);
            var lweights = new Moment[layerObj.NeuronCount];
            m_bias[layer] = lweights;
            for (var w = 0; w < lweights.Length; w++) {
                lweights[w] = new Moment{ First = 0d, Second = 0d };
            }
        }
    }

    // TODO double check the below logic. Make NULL safe (I mean shouldnt be an issue since initialize should set everything up... but could be if I forget to call it)
    // Dereference of a possibly null reference.
    #pragma warning disable CS8602
    public double UpdateLearningRate(double baseLearningRate, double gradient, int layer, int neuron, int weight) {
        timestep++; // Increment the time step (updated ALL times... should it only be updated once per layer, or once per epoch?)

        // Update biased first moment estimate
        var cache = m_weights[layer][neuron];
        cache[weight].First = Beta1 * cache[weight].First + (1 - Beta1) * gradient;

        // Update biased second moment estimate
        cache[weight].Second = Beta2 * cache[weight].Second + (1 - Beta2) * gradient * gradient;

        // Bias correction
        double mHat = cache[weight].First / (1 - Math.Pow(Beta1, timestep));
        double vHat = cache[weight].Second / (1 - Math.Pow(Beta2, timestep));

        // Adjusted learning rate
        return baseLearningRate * mHat / (Math.Sqrt(vHat) + double.Epsilon);
    }

    public double UpdateLearningRate(double baseLearningRate, double gradient, int layer, int neuron) {
        timestep++; // Increment the time step (updated ALL times... should it only be updated once per layer)

        // Update biased first moment estimate
        var cache = m_bias[layer];
        cache[neuron].First = Beta1 * cache[neuron].First + (1 - Beta1) * gradient;

        // Update biased second moment estimate
        cache[neuron].Second = Beta2 * cache[neuron].Second + (1 - Beta2) * gradient * gradient;

        // Bias correction
        double mHat = cache[neuron].First / (1 - Math.Pow(Beta1, timestep));
        double vHat = cache[neuron].Second / (1 - Math.Pow(Beta2, timestep));

        // Adjusted learning rate
        return baseLearningRate * mHat / (Math.Sqrt(vHat) + double.Epsilon);
    }
    // Dereference of a possibly null reference.
    #pragma warning restore CS8602 
}