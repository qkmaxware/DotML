using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer that performs batch normalization. Each channel is normalized across all batches.
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
public class BatchNorm : FeedforwardNetworkLayer, INormalizationLayer {

    private float running_mean_momentum = 0.9f;
    public Vec<float> RunningMean;
    private float running_variance_momentum = 0.9f;
    public Vec<float> RunningVariance;

    /// <summary>
    /// Normalization scaling factor
    /// </summary>
    public Vec<float> Gammas;

    /// <summary>
    /// Normalization shifting offset
    /// </summary>
    public Vec<float> Betas;

    public BatchNorm(Shape3D input_size, float mean_momentum = 0.9f, float variance_momentum = 0.9f) {
        this.InputShape = input_size;
        this.OutputShape = input_size;

        this.running_mean_momentum = mean_momentum;
        this.running_variance_momentum = variance_momentum;

        var rmean = new float[input_size.Channels];
        Array.Fill(rmean, 0.0f);
        this.RunningMean = rmean;
        var rvariance = new float[input_size.Channels];
        Array.Fill(rvariance, 1.0f);
        this.RunningVariance = rvariance;

        this.Gammas = new Vec<float>(input_size.Channels, 1.0f);
        this.Betas = new Vec<float>(input_size.Channels, 0);
    }

    public override FeatureSet<float> EvaluateSync(FeatureSet<float> channels) {
        // Basically this is used when not training
        // Not sure if this is how to do it when there is no batching (like during evaluation rather than training)
        return EvaluateSync(new BatchedFeatureSet<float>(channels))[0];
    }

    public void ComputeMeansAndVariances(BatchedFeatureSet<float> features, out float[] mean_vec, out float[] variance_vec) {
        if (IsInference) {
            mean_vec = (float[])this.RunningMean;
            variance_vec = (float[])this.RunningVariance;
            return;
        } 

        mean_vec = new float[features.Channels];
        variance_vec = new float[features.Channels];
        for (var channelIndex = 0; channelIndex < variance_vec.Length; channelIndex++) {
            float mean = 0.0f;
            float m2 = 0.0f;
            int count = 0;

            foreach (var featureSet in features) {
                var matrix = featureSet[channelIndex];
                var rows = matrix.Rows;
                var cols = matrix.Columns;
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        count++;
                        float value = matrix[i, j];
                        var delta = value - mean;
                        mean += delta / count;
                        var delta2 = value - mean;
                        m2 += delta * delta2;
                    }
                }
            }

            mean_vec[channelIndex] = mean;
            variance_vec[channelIndex] = m2 / count;
        }
    }

    public override BatchedFeatureSet<float> EvaluateSync(BatchedFeatureSet<float> features) {
        // Compute mean and variance across the whole batch per channel
        this.ComputeMeansAndVariances(features, out var means, out var variances);

        // Perform running mean/variance update step
        if (this.IsInference && features.Batches > 2) {
            for (var channelIndex = 0; channelIndex < variances.Length; channelIndex++) {
                var mean = means[channelIndex];
                var variance = variances[channelIndex];
                
                ((float[])RunningMean)[channelIndex] = running_mean_momentum * mean + (1 - running_mean_momentum) * RunningMean[channelIndex];
                ((float[])RunningVariance)[channelIndex] = running_variance_momentum * variance + (1 - running_variance_momentum) * RunningVariance[channelIndex];
            }
        }

        // When run with a batch size of > 1
        var results = new FeatureSet<float>[features.Batches];
        Parallel.For(0, features.Batches, (batchIndex) => {
            var featureSet = features[batchIndex];
            results[batchIndex] = new FeatureSet<float>(
                featureSet.Select((matrix, channelIndex) => {
                    // Compute mean and variance for the channel
                    var mean = means[channelIndex];
                    var variance = variances[channelIndex];

                    // Normalize the channel using mean and variance
                    var v = MathF.Sqrt(variance + 1e-8f);
                    var normalizedMatrix = matrix.Transform(x => (x - mean)  / v);

                    // Apply scaling (gamma) and shifting (beta)
                    // 0, 2, 1, 1
                    var gamma = Gammas[channelIndex];
                    var beta =  Betas[channelIndex];
                    normalizedMatrix.Apply((x) => x * gamma + beta); // output = output .* gamma // output = output + beta    

                    return normalizedMatrix;
                }).ToArray()
            );
        });
        return new BatchedFeatureSet<float>(results);
    }

    const float epsilon = 1e-8f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SumAll(Matrix<float>[,] matrices, int channel_to_sum) {
        float sum = 0;
        var batches = matrices.GetLength(0);
        for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
            var matrix = matrices[batchIndex, channel_to_sum];
            var rows = matrix.Rows;
            var cols = matrix.Columns;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    sum += matrix[i, j];
                }
            }
        }
        return sum;
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        // Terminology
        // x    is input
        // xHat is output before scaling and shifting
        // y    is output after scaling and shifting
        // g    is a scaling value from the gamma matrix
        // b    is a shifting value from the beta matrix
        // u    is the mean 
        // o2   is the variance
        var errors = args.OutputErrors;
        var batches  = args.OutputErrors.Batches;
        var channels = args.OutputErrors.Channels;
        var rows = args.OutputErrors.Rows;
        var columns = args.OutputErrors.Columns;
        var one_over_features = 1.0f / (rows * columns);
    
        Vec<float> gradient_betas = new Vec<float>(channels);
        Vec<float> gradient_gammas = new Vec<float>(channels);
        var input_gradients = new Matrix<float>[batches][];
        for (var batch = 0; batch < batches; batch++) {
            input_gradients[batch] = new Matrix<float>[channels];
        }

        // Compute the mean and variances for for the inputs across each batch
        float[] mean_per_channel;
        float[] variance_per_channel;
        ComputeMeansAndVariances(args.InputBatch, out mean_per_channel, out variance_per_channel);

        // Gradient of L with respect to beta
        // dL/dB = Sum_b( dL/dY )
        for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
            var feats = errors[batchIndex];
            for (var channelIndex = 0; channelIndex < channels; channelIndex++) {
                var dL_dY = feats[channelIndex];
                gradient_betas[channelIndex] += dL_dY.Sum();
            }
        }

        // Gradient of L with respect to gamma
        // dL/dG = Sum_b( dL/dY * xHat )
        for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
            var xs = args.InputBatch[batchIndex];
            var ys = args.OutputBatch[batchIndex];
            var err = args.OutputErrors[batchIndex];
            for (var channelIndex = 0; channelIndex < channels; channelIndex++) {
                var dL_dY = err[channelIndex];
                /*var xHat = xs[channelIndex].Transform(x => 
                    (x - mean_per_channel[channelIndex]) / Math.Sqrt(variance_per_channel[channelIndex] + epsilon)
                );*/
                var xHat = ys[channelIndex].Transform(x => (x - Betas[channelIndex]) / (Gammas[channelIndex] + epsilon));
                var sum = dL_dY.HadamardWith(xHat).Sum();
                gradient_gammas[channelIndex] += sum;
            }
        }

        // Gradient of L with respect to xHat
        // dL/dXHat = dL/dY * gamma
        var dL_dXHats = new Matrix<float>[batches, channels];
        for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
            var xs = args.InputBatch[batchIndex];
            var ys = args.OutputBatch[batchIndex];
            var err = args.OutputErrors[batchIndex];

            for (var channelIndex = 0; channelIndex < channels; channelIndex++) {
                var dL_dY = err[channelIndex];
                var gamma = Gammas[channelIndex];
                dL_dXHats[batchIndex, channelIndex] = dL_dY.Transform(x => x * gamma);
            }
        }

        // Gradient of L with respect to x
        for (var channelIndex = 0; channelIndex < channels; channelIndex++) {
            var var = variance_per_channel[channelIndex];
            var mean = mean_per_channel[channelIndex];
            var m = batches * rows * columns; 
            var _m = 1.0f / m;
            var var_plus_epsilon = var + epsilon;
            var inv_var_plus_epsilon = 1.0f / var_plus_epsilon;
            var sqrt = MathF.Sqrt(var_plus_epsilon);
            var _sqrt = 1.0f / sqrt;
            var sum_all_dl_dxhat = SumAll(dL_dXHats, channelIndex); //loss_wrt_xhats.SelectMany(xhat => xhat).Sum();
            var term2_scalar = -sum_all_dl_dxhat / (m * sqrt);

            var sum_all_dxHat_and_x = 0.0f;
            {
                var x_count = batches;
                for (var i = 0; i < x_count; i++) {
                    var xhat = dL_dXHats[i, channelIndex].AsSpan();
                    var x = args.InputBatch[i, channelIndex].AsSpan();
                    var el_count = xhat.Length;
                    for (var j = 0; j < el_count; j++) {
                        sum_all_dxHat_and_x += xhat[j] * (x[j] - mean) * inv_var_plus_epsilon;
                    }
                }    
            }

            for (var batch = 0; batch < batches; batch++) {
                var x = args.InputBatch[batch, channelIndex];
                var y = args.OutputBatch[batch, channelIndex];
                var dY = args.OutputErrors[batch, channelIndex];
                var dxHat = dL_dXHats[batch, channelIndex];
            
                var term1 = dxHat * _sqrt;

                var term3 = x.Transform((xi) => -(_m * _sqrt) * (xi - mean) * sum_all_dxHat_and_x);

                input_gradients[batch][channelIndex] = term1.ElementWise(term3, (t1, t3) => t1 + term2_scalar + t3); // term1 + term2 + term3 | term1,term2 are matrices, term2 is a scalar value 
            }
        }

        // Return the gradients
        return new BackpropagationReturns(
            new BatchedFeatureSet<float>(input_gradients.Select(x => new FeatureSet<float>(x)).ToArray()),
            new Gradients (
                this.Gammas,
                this.Betas,
                gradient_gammas,
                gradient_betas
            )
        );
    }

    public override void SubtractGradients(LayerGradients? gradients) {
        if (gradients is null || gradients is not Gradients grads)
            throw new ArgumentException(nameof(gradients));

        this.Gammas = this.Gammas - grads.GammaGradients;
        this.Betas = this.Betas - grads.BetaGradients;
    }

    public override void Initialize(IInitializer initializer) {
        // No need to initialize any weights
    }

    public override int TrainableParameterCount() {
        return this.Gammas.Dimensionality + this.Betas.Dimensionality;
    }

    /// <summary>
    /// Number of un-trainable parameters in this layer
    /// </summary>
    /// <returns>Number of un-trainable parameters</returns>
    public override int UnTrainableParameterCount() => this.RunningVariance.Dimensionality + this.RunningMean.Dimensionality;

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => visitor.Visit(this, args);

    public override T Visit<T>(ILayerOutputVisitor<T> visitor) => visitor.Visit(this);

    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);

    public class Gradients : LayerGradients {
        private Vec<float> Gammas;
        private Vec<float> Betas;
        public Vec<float> GammaGradients;
        public Vec<float> BetaGradients;

        public Gradients(Vec<float> gamma, Vec<float> beta, Vec<float> gammagrad, Vec<float> betagrad) {
            this.Gammas = gamma;
            this.Betas = beta;
            this.GammaGradients = gammagrad;
            this.BetaGradients = betagrad;
        }

        public override void Clip(float weight_threshold, float bias_threshold) {
            ClipVector(GammaGradients, weight_threshold);
            ClipVector(BetaGradients, weight_threshold);
        }

        public override void Apply(GradientTransformationHandler handler) {
            int index = 0;
            for (var i = 0; i < this.GammaGradients.Dimensionality; i++) {
                handler(index, Gammas[i], GammaGradients[i]);
                index++;
            }

            for (var i = 0; i < this.BetaGradients.Dimensionality; i++) {
                handler(index, Betas[i], BetaGradients[i]);
                index++;
            }
        }
    }
}