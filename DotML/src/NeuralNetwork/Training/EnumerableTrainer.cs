using System.Collections;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network.Training;

#region Enumerable
/// <summary>
/// Simple Neural Network trainer based on backpropagation
/// </summary>
/// <typeparam name="TNetwork">type of network to train</typeparam>
public class EnumerableBackpropagationTrainer<TNetwork>
    : IEnumerableTrainer<TNetwork>
where TNetwork : ILayeredNeuralNetwork<ILayerWithNeurons>
{
    /// <summary>
    /// Create a new Neural Network trainer
    /// </summary>
    public EnumerableBackpropagationTrainer() {}

    /// <summary>
    /// Number of epochs (default: 250)
    /// </summary>
    public int Epochs {get; set;} = 250;

    /*/// <summary>
    /// Number of training items per batch (default: 1)
    /// </summary>
    //public int BatchSize {get; set;} = 1;*/

    /// <summary>
    /// Flag to indicate if training should stop before the MaxEpochs has been reached if the network has achieved the desired accuracy (default: false)
    /// </summary>
    public bool EarlyStop {get; set;} = false;

    /// <summary>
    /// The accuracy that is used as a condition to stop training if EarlyStop is set to true (default: 0.1)
    /// </summary>
    public double EarlyStopAccuracy {
        get => _earlyStopAccuracy;
        set {
            _earlyStopAccuracy = Math.Max(0, value);
        }
    }
    private double _earlyStopAccuracy = 0.1;

    /// <summary>
    /// The loss function used in network accuracy evaluation (default: MSE)
    /// </summary>
    public LossFunction LossFunction {get; set;} = LossFunctions.MeanSquaredError;

    /// <summary>
    /// Learning rate for changes to weights/biases, usually between 0.0001 and 0.1. (default: 0.1)
    /// </summary>
    public double LearningRate {get; set;} = 0.1;
    /// <summary>
    /// Strategy for learning rate adjusting during training. (default: ConstantRate)
    /// </summary>
    public ILearningRateOptimizer LearningRateOptimizer {get; set;} = new ConstantRate();
    /// <summary>
    /// Momentum factor applied to changes to weights & biases based on previous changes set only. Use WeightMomentumFactor and BiasMomentumFactor to apply different momentum factors to weights and biases accordingly.
    /// </summary>
    public double MomentumFactor {
        set {
            WeightMomentumFactor = value;
            BiasMomentumFactor = value;
        }
    }
    /// <summary>
    /// Momentum factor applied to changes to weights based on previous changes. (default: 0)
    /// </summary>
    public double WeightMomentumFactor {get; set;} = 0.0;
    /// <summary>
    /// Momentum factor applied to changes to biases based on previous changes. (default: 0)
    /// </summary>
    public double BiasMomentumFactor {get; set;} = 0.0;

    /// <summary>
    /// Enable to perform clipping of gradients with magnitudes larger than the GradientClipThreshold value. (default: false)
    /// </summary>
    public bool EnableGradientClipping {get; set;} = false;
    /// <summary>
    /// Threshold to compare gradients against when EnableGradientClipping is set. Values between 1.0 and 5.0 are common. (default 5.0)
    /// </summary>
    public double GradientClipThreshold {
        get => _gradientClipThreshold;
        set {
            _gradientClipThreshold = Math.Max(0, value);
        }
    }
    private double _gradientClipThreshold = 0.5;

    /// <summary>
    /// The regularization function used when updating weights. (default: Null)
    /// </summary>
    public RegularizationFunction? Regularization {get; set;} = null;

    /// <summary>
    /// Network initialization strategy (defaults: NormalXavierInitialization)
    /// </summary>
    public IInitializer NetworkInitializer {get; set;} = new NormalXavierInitialization();

    /// <summary>
    /// Train the Neural Network by sampling the training data using the provided sequencer
    /// </summary>
    /// <param name="enumerator">training data sequence</param>
    /// <param name="network">neural network to trail</param>
    /// <param name="epochs">number of epochs to train for</param>
    /// <param name="learningRate">learning rate for backpropagation</param>
    public IEpochEnumerator<TNetwork> EnumerateTraining(TNetwork network, IEnumerator<TrainingPair> dataset, IEnumerator<TrainingPair> validation) {
        return new BackpropagationEnumerator<TNetwork> (
            network:                    network, 
            enumerator:                 dataset, 
            validator:                  validation, 
            //batch_size:                 BatchSize,
            init:                       NetworkInitializer, 

            epochs:                     Epochs, 
            earlyStop:                  EarlyStop,
            earlyStopAccuracy:          EarlyStopAccuracy,
            earlyStopAccuracyFunction:  LossFunction,

            learningRateOptimizer:      LearningRateOptimizer,
            learningRate:               LearningRate, 
            weightMomentumFactor:       WeightMomentumFactor,
            biasMomentumFactor:         BiasMomentumFactor,
            gradientClipping:           EnableGradientClipping,
            clipThreshold:              GradientClipThreshold,   

            regularization:             Regularization
        );
    }

    /// <summary>
    /// Train the Neural Network by sampling the training data using the provided sequencer
    /// </summary>
    /// <param name="enumerator">training data sequence</param>
    /// <param name="network">neural network to trail</param>
    /// <param name="epochs">number of epochs to train for</param>
    /// <param name="learningRate">learning rate for backpropagation</param>
    public void Train(TNetwork network, IEnumerator<TrainingPair> dataset, IEnumerator<TrainingPair> validation) {
        EnumerateTraining(network, dataset, validation).MoveToEnd();
    }
}
#endregion

#region Enumerator
/// <summary>
/// Simple enumerator for a backpropagation trainer
/// </summary>
/// <typeparam name="TNetwork">type of network being trained</typeparam>
public class BackpropagationEnumerator<TNetwork> : IEpochEnumerator<TNetwork> where TNetwork : ILayeredNeuralNetwork<ILayerWithNeurons> {
    /// <summary>
    /// Number of training items per batch
    /// </summary>
    public int BatchSize {get; private set;}

    /// <summary>
    /// Maximum number of epochs to perform
    /// </summary>
    public int MaxEpochs {get; private set;}

    /// <summary>
    /// Flag to indicate if training should stop before the MaxEpochs has been reached if the network has achieved the desired accuracy.
    /// </summary>
    public bool EarlyStop {get; private set;}

    /// <summary>
    /// The accuracy that is used as a condition to stop training if EarlyStop is set to true
    /// </summary>
    public double EarlyStopAccuracy {get; private set;}

    /// <summary>
    /// The loss function used in network accuracy evaluation
    /// </summary>
    public LossFunction LossFunction {get; private set;}

    /// <summary>
    /// Learning rate for changes to weights/biases.
    /// </summary>
    public double LearningRate {get; private set;}
    /// <summary>
    /// Strategy for learning rate adjusting during training
    /// </summary>
    public ILearningRateOptimizer LearningRateOptimizer {get; private set;}

    /// <summary>
    /// Momentum factor applied to changes to weights based on previous changes
    /// </summary>
    public double WeightMomentumFactor {get; private set;} = 0.0;

    /// <summary>
    /// Momentum factor applied to changes to biases based on previous changes
    /// </summary>
    public double BiasMomentumFactor {get; private set;} = 0.0;
    /// <summary>
    /// Enable to perform clipping of gradients with magnitudes larger than the GradientClipThreshold value.
    /// </summary>
    public bool EnableGradientClipping {get;  private set;}
    /// <summary>
    /// Threshold to compare gradients against when EnableGradientClipping is set.
    /// </summary>
    public double GradientClipThreshold {get; private set;}

    /// <summary>
    /// Current training epoch
    /// </summary>
    public int CurrentEpoch {get; private set;}
    private int updateTimestep = 0;

    /// <summary>
    /// Current status of the neural network being trained
    /// </summary>
    public TNetwork Current {get; private set;}
    object IEnumerator.Current => Current;

    /// <summary>
    /// Network initialization strategy
    /// </summary>
    public IInitializer NetworkInitializer {get; set;}

    // Standard Training Implementation --------------------
    private IEnumerator<TrainingPair> enumerator;
    private IEnumerator<TrainingPair> validator;
    //private List<TrainingPair> batch;
    private double[][] deltas;
    //private double[][] input_weight_gradients;
    //private double[] input_bias_gradients;

    /// <summary>
    /// Get the computed input weight gradients for the given input neuron
    /// </summary>
    /// <param name="neuron">input neuron index</param>
    /// <returns>weight gradients</returns>
    //public Vec<double> GetInputWeightGradients(int neuron) => Vec<double>.Wrap(input_weight_gradients[neuron]);
    /// <summary>
    /// Get the computed input bias gradients for the given input neuron
    /// </summary>
    /// <param name="neuron">input neuron index</param>
    /// <returns>bias gradients</returns>
    //public Vec<double> GetInputBiasGradients() => Vec<double>.Wrap(input_bias_gradients);

    // Regularization Implementation ----------------------
    private RegularizationFunction? regularization;

    // Momentum Implementation ----------------------------
    private double[][][] weightMomentum;
    private double[][] biasMomentum;

    // Early Stop Implementation --------------------------
    private double? current_loss;
    
    // Adaptive Learning Rate Implementation -------------- TODO

    public BackpropagationEnumerator (
        TNetwork network, 
        IEnumerator<TrainingPair> enumerator, 
        IEnumerator<TrainingPair> validator, 
        //int batch_size,
        IInitializer init, 

        int epochs, bool earlyStop, double earlyStopAccuracy, LossFunction earlyStopAccuracyFunction,
        
        ILearningRateOptimizer learningRateOptimizer,
        double learningRate, double weightMomentumFactor, double biasMomentumFactor,
        bool gradientClipping, double clipThreshold,
        RegularizationFunction? regularization
    ) {
        // Safety checks
        epochs = epochs == int.MaxValue ? int.MaxValue - 1 : epochs;
        learningRate = learningRate == 0.0 ? double.Epsilon : learningRate;
        WeightMomentumFactor = Math.Max(0, weightMomentumFactor);
        BiasMomentumFactor = Math.Max(0, biasMomentumFactor);

        // Initialization
        MaxEpochs = epochs;
        LearningRate = learningRate;
        Current = network;
        CurrentEpoch = 0;
        
        //this.BatchSize = Math.Max(1, batch_size);
        //this.batch = new List<TrainingPair>(this.BatchSize);
        this.EnableGradientClipping = gradientClipping;
        this.GradientClipThreshold = Math.Max(0, clipThreshold);
        this.EarlyStop = earlyStop;
        this.EarlyStopAccuracy = earlyStopAccuracy;
        this.LossFunction = earlyStopAccuracyFunction;
        this.enumerator = enumerator;
        this.validator = validator;
        this.NetworkInitializer = init;
        this.regularization = regularization;
        this.LearningRateOptimizer = learningRateOptimizer;

        // Additional heap allocations
        var layer_count = network.LayerCount;
        deltas = new double[layer_count][];
        weightMomentum = new double[layer_count][][];
        biasMomentum = new double[layer_count][];
        for (var layerIndex = 0; layerIndex < layer_count; layerIndex++) {
            var layer = network.GetLayer(layerIndex);
            var neurons = layer.NeuronCount;
            deltas[layerIndex] = new double[neurons];
            var wm = new double[neurons][];
            weightMomentum[layerIndex] = wm;
            for (var neuronIndex = 0; neuronIndex < neurons; neuronIndex++) {
                wm[neuronIndex] = new double[layer.GetNeuron(neuronIndex).Weights.Length];
            }
            biasMomentum[layerIndex] = new double[neurons];
        }

        /*input_weight_gradients = new double[network.InputCount][];
        input_bias_gradients = new double[network.InputCount];
        var input_layer = network.GetFirstLayer();
        for (var neuron_index = 0; neuron_index < network.InputCount; neuron_index++) {
            var neuron = input_layer.GetNeuron(neuron_index);
            input_weight_gradients[neuron_index] = new double[neuron.Weights.Length];
        }*/

        // Prepare network for training
        Reset();
    }

    public void Dispose() { /* No need to do anything */ }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double getInputWeightOrDefault(INeuron neuron, int weight)  {
        var weights = neuron.Weights;
        if (weight < 0 || weight >= weights.Length)
            return default(double);
        return weights[weight];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void backpropagate(ILayerWithNeurons layer, double[] layerDeltas, Vec<double> targets) {
        var neuron_count = layer.NeuronCount;
        var outputs = layer.GetLastOutputs();

        for (var neuronIndex = 0; neuronIndex < neuron_count; neuronIndex++) {
            INeuron neuron = layer.GetNeuron(neuronIndex);
            var output = outputs[neuronIndex];
            var target = targets[neuronIndex];

            var error = target - output;
            var activation = neuron.ActivationFunction ?? Identity.Instance;
            var slope = activation.InvokeDerivative(output);

            layerDeltas[neuronIndex] = slope * error;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void backpropagate(ILayerWithNeurons layer, double[] layerDeltas, ILayerWithNeurons next_layer, double[] next_layerDeltas) {
        var neuron_count = layer.NeuronCount;
        var nextlayer_neuron_count = next_layer.NeuronCount;
        var outputs = layer.GetLastOutputs();
        
        for (var neuronIndex = 0; neuronIndex < neuron_count; neuronIndex++) {
            INeuron neuron = layer.GetNeuron(neuronIndex);
            var output = outputs[neuronIndex];

            var error = 0.0;
            for (var nextLayerNeuronIndex = 0; nextLayerNeuronIndex < nextlayer_neuron_count; nextLayerNeuronIndex++) {
                INeuron nextLayerNeuron = next_layer.GetNeuron(nextLayerNeuronIndex);
                error += next_layerDeltas[nextLayerNeuronIndex] * getInputWeightOrDefault(nextLayerNeuron, neuronIndex);
            }
            var activation = neuron.ActivationFunction ?? Identity.Instance;
            var slope = activation.InvokeDerivative(output);

            layerDeltas[neuronIndex] = slope * error;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double clip_gradient(double gradient) {
        // If clipping is enabled AND we are above the clipping threshold then clip, otherwise just return the gradient as is
        return 
            EnableGradientClipping && Math.Abs(gradient) > GradientClipThreshold
            ? Math.Sign(gradient) * GradientClipThreshold
            : gradient;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void updateWeightsAndBias(int timestep, int parameters_start, bool isInputLayer, Vec<double> inputs, double learningRate, int layerIndex, ILayerWithNeurons layer, double[] deltas, double[][] weight_momentum, double[] bias_momentum) {
        var neuron_count = layer.NeuronCount;

        var weight_parameters = layer.InputCount * layer.OutputCount;
        var bias_parameters = layer.NeuronCount;
        var weight_parameter_index = 0;
        var bias_parameters_index = 0;

        for (int neuronIndex = 0; neuronIndex < neuron_count; neuronIndex++) {
            INeuron neuron = layer.GetNeuron(neuronIndex);
            var weights = neuron.Weights;
            var momentums = weight_momentum[neuronIndex];

            var weight_count = weights.Length;
            var gradient = deltas[neuronIndex];
            for (int weightIndex = 0; weightIndex < weight_count; weightIndex++) {
                var current_weight = weights[weightIndex];
                var combinedGradient = clip_gradient(gradient * inputs[weightIndex]);

                var weightAdjustedLearningRate = LearningRateOptimizer.UpdateLearningRate(timestep, learningRate, combinedGradient, weight_parameter_index++);
                var w_momentum = weightAdjustedLearningRate * combinedGradient + WeightMomentumFactor * momentums[weightIndex];
                momentums[weightIndex] = w_momentum;

                var next_weight = current_weight + w_momentum; // (w_momentum already includes this rateTimeDelta * inputs[j]);
                if (regularization is not null) {
                    next_weight += regularization.Invoke(current_weight);
                }
                weights[weightIndex] = next_weight; 
                if (isInputLayer) {
                    //input_weight_gradients[neuronIndex][weightIndex] = combinedGradient;
                }
            }

            var biasGradient = clip_gradient(gradient);
            var biasAdjustedLearningRate = LearningRateOptimizer.UpdateLearningRate(timestep, learningRate, biasGradient, weight_parameters + bias_parameters_index++);
            var b_momentum = (biasAdjustedLearningRate * biasGradient) + BiasMomentumFactor * bias_momentum[neuronIndex];
            bias_momentum[neuronIndex] = b_momentum;
            var next_bias = neuron.Bias + b_momentum; // (b_momentum already includes this rateTimeDelta);
            neuron.Bias = next_bias;
            if (isInputLayer) {
                //input_bias_gradients[neuronIndex] = biasGradient;
            }
        }
    }

    /// <summary>
    /// Advance the training by one epoch
    /// </summary>
    /// <returns>true if the training has been advanced, false if we've run out of epochs or have already finished training the network.</returns>
    public bool MoveNext() {
        // Stop condition 
        if (CurrentEpoch >= MaxEpochs) { 
            return false;
        }

        // Backpropagation Core
        #region Backpropagation
        var layer_count = Current.LayerCount;
        enumerator.Reset();

        /*
        // Setup initial batch
        batch.Clear();
        for (var i = 0; i < BatchSize; i++) {
            if (enumerator.MoveNext()) {
                batch.Add(enumerator.Current);
            }
        }
        // Loop over batches
        while (batch.Count > 0) {
            // Feed-forward step
            var layer_outputs = new Vec<double>[layer_count][batch.Count];
            foreach (var data in batch) {
                var input               = data.Input;
                var expected            = data.Output;
                var actual              = Current.PredictSync(input);
                for (var l = 0; l < layer_count; l++) {
                    layer_outputs[l][batch] = Current.GetLayer(i).GetLastOutputs();
                }
            }

            // Errors on output layer
            backpropagate(Current.GetOutputLayer(), deltas[^1], expected);

            // Backpropagation through hidden layers
            for (var layerIndex = layer_count - 2; layerIndex >= 0; layerIndex--) {
                backpropagate(Current.GetLayer(layerIndex), deltas[layerIndex], Current.GetLayer(layerIndex + 1), deltas[layerIndex + 1]);
            }

            // Update weights and biases
            for (int layerIndex = layer_count - 1; layerIndex >= 0; layerIndex--) {
                Vec<double> inputs = (layerIndex == 0) ? batch[0].Input : Current.GetLayer(layerIndex - 1).GetLastOutputs();
                updateWeightsAndBias(inputs, LearningRate, layerIndex, Current.GetLayer(layerIndex), deltas[layerIndex], weightMomentum[layerIndex], biasMomentum[layerIndex]);
            }

            // Setup next batch
            batch.Clear();
            for (var i = 0; i < BatchSize; i++) {
                if (enumerator.MoveNext()) {
                    batch.Add(enumerator.Current);
                }
            }
        }*/

        while (enumerator.MoveNext()) {
            var data                = enumerator.Current;
            
            // Feed-forward step
            var input               = data.Input;
            var expected            = data.Output;
            var actual              = Current.PredictSync(input);

            // Errors on output layer
            backpropagate(Current.GetOutputLayer(), deltas[^1], expected);

            // Backpropagation through hidden layers
            for (var layerIndex = layer_count - 2; layerIndex >= 0; layerIndex--) {
                backpropagate(Current.GetLayer(layerIndex), deltas[layerIndex], Current.GetLayer(layerIndex + 1), deltas[layerIndex + 1]);
            }

            // Update weights and biases
            var parameter_offset = 0;
            for (int layerIndex = layer_count - 1; layerIndex >= 0; layerIndex--) {
                bool isInputLayer = layerIndex == 0;
                Vec<double> inputs = (isInputLayer) ? input : Current.GetLayer(layerIndex - 1).GetLastOutputs();
                var layer = Current.GetLayer(layerIndex);
                updateWeightsAndBias(updateTimestep, parameter_offset, isInputLayer, inputs, LearningRate, layerIndex, layer, deltas[layerIndex], weightMomentum[layerIndex], biasMomentum[layerIndex]);
                parameter_offset += layer.TrainableParameterCount();
            }
            updateTimestep++;
        }
        #endregion

        // Early Stop & Accuracy Test
        #region Early Stop
        var validation = this.validator ?? this.enumerator;
        if (EarlyStop && validation is not null) {
            validation.Reset();

            var loss = 0.0;
            var count = 0;
            while (validation.MoveNext()) {
                var set = validation.Current;
                var predicted = Current.PredictSync(set.Input);
                loss = Math.Max(loss, LossFunction(predicted, set.Output)); // loss + LossFunction(predicted, set.Output) for avg, currently max
                count++;
            }
            if (count > 0)
                current_loss = loss; // / count; // Should this be max or avg ? currently max
        }
        #endregion

        CurrentEpoch++;

        return 
            CurrentEpoch < MaxEpochs                                                            // Epoch reached condition
        && (!EarlyStop || (current_loss.HasValue && current_loss.Value > EarlyStopAccuracy))            // Early stop accuracy reached
        ;
    }

    /// <summary>
    /// Reset training
    /// </summary>
    public void Reset() {
        CurrentEpoch = 0;
        updateTimestep = 0;
        //this.batch.Clear();
        this.current_loss = null;
        Current.Initialize(NetworkInitializer);
        //NetworkInitializer.InitializeWeights(Current);
        //NetworkInitializer.InitializeBiases(Current);
        LearningRateOptimizer.Initialize(Current);

        initMomentum();
    }

    private void initMomentum() {
        // li is layer_index, ni = neuron_index, wi = weight_index shortening here for simplicity of typing
        // Weight momentum
        for (var li = 0; li < weightMomentum.Length; li++) {
            var l = weightMomentum[li];
            for (var ni = 0; ni < l.Length; ni++) {
                var n = l[ni];
                for (var wi = 0; wi < n.Length; wi++) {
                    n[wi] = 0.0;
                }
            }
        }

        // Bias momentum
        for (var li = 0; li < biasMomentum.Length; li++) {
            var l = biasMomentum[li];
            for (var ni = 0; ni < l.Length; ni++) {
                l[ni] = 0.0;
            }
        }
    }
}
#endregion