using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer that performs batch normalization 
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
[Untested]
public class BatchNorm : FeedforwardNetworkLayer {

    private double running_mean_momentum = 0.9;
    public Vec<double> RunningMean;
    private double running_variance_momentum = 0.9;
    public Vec<double> RunningVariance;

    /// <summary>
    /// Normalization scaling factor
    /// </summary>
    [JsonIgnore]
    public Matrix<double>[] Gammas {get; set;}
    /// <summary>
    /// Normalization shifting offset
    /// </summary>
    [JsonIgnore]
    public Matrix<double>[] Betas {get; set;}

    public BatchNorm(Shape3D input_size, double mean_momentum = 0.9, double variance_momentum = 0.9) {
        this.InputShape = input_size;
        this.OutputShape = input_size;

        this.running_mean_momentum = mean_momentum;
        this.running_variance_momentum = variance_momentum;

        var rmean = new double[input_size.Channels];
        Array.Fill(rmean, 0.0);
        this.RunningMean = rmean;
        var rvariance = new double[input_size.Channels];
        Array.Fill(rvariance, 1.0);
        this.RunningVariance = rvariance;

        this.Gammas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Gammas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 1.0);
        this.Betas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Betas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 0.0);
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        // Basically this is used when not training
        // Not sure if this is how to do it when there is no batching (like during evaluation rather than training)
        return EvaluateSync(new BatchedFeatureSet<double>(channels))[0];
    }

    public void ComputeMeansAndVariances(BatchedFeatureSet<double> features, out double[] mean_vec, out double[] variance_vec) {
        if (IsInference || features.Batches < 2) {
            mean_vec = (double[])this.RunningMean;
            variance_vec = (double[])this.RunningVariance;
        } else {
            var means = new double[features.Channels];
            var variances = new double[features.Channels];
            for (var channelIndex = 0; channelIndex < variances.Length; channelIndex++) {
                double sum = 0.0;
                double sumSq = 0.0;
                int count = 0;

                foreach (var featureSet in features) {
                    var matrix = featureSet[channelIndex];
                    var rows = matrix.Rows;
                    var cols = matrix.Columns;
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) {
                            double value = matrix[i, j];
                            sum += value;
                            sumSq += value * value;
                            count++;
                        }
                    }
                }

                count = Math.Max(count, 1); // Avoid division by zero
                double mean = sum / count;
                double variance = (sumSq / count) - (mean * mean);

                means[channelIndex] = mean;
                variances[channelIndex] = variance;
            }

            mean_vec = means;
            variance_vec = variances;
        }
    }

    public override BatchedFeatureSet<double> EvaluateSync(BatchedFeatureSet<double> features) {
        // Compute mean and variance across the whole batch per channel
        this.ComputeMeansAndVariances(features, out var means, out var variances);

        // Perform running mean/variance computation
        if (this.IsInference && features.Batches > 2) {
            for (var channelIndex = 0; channelIndex < variances.Length; channelIndex++) {
                var mean = means[channelIndex];
                var variance = variances[channelIndex];
                
                ((double[])RunningMean)[channelIndex] = running_mean_momentum * mean + (1 - running_mean_momentum) * RunningMean[channelIndex];
                ((double[])RunningVariance)[channelIndex] = running_variance_momentum * variance + (1 - running_variance_momentum) * RunningVariance[channelIndex];
            }
        }

        // When run with a batch size of > 1
        var results = new FeatureSet<double>[features.Batches];
        Parallel.For(0, features.Batches, (batchIndex) => {
            var featureSet = features[batchIndex];
            results[batchIndex] = new FeatureSet<double>(
                featureSet.Select((matrix, channelIndex) => {
                    // Compute mean and variance for the channel
                    var mean = means[channelIndex];
                    var variance = variances[channelIndex];

                    // Normalize the channel using mean and variance
                    var normalizedMatrix = matrix.Transform(x => (x - mean)  / Math.Sqrt(variance + 1e-8));

                    // Apply scaling (gamma) and shifting (beta)
                    normalizedMatrix.HadamardWithInplace(Gammas[channelIndex]); // output = output .* gamma
                    normalizedMatrix.AddWithInplace(Betas[channelIndex]); // output = output + beta     

                    return normalizedMatrix;
                }).ToArray()
            );
        });
        return new BatchedFeatureSet<double>(results);
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

        var batches  = args.OutputErrors.Batches;
        var one_over_batches = 1.0 / batches;
        var channels = args.OutputErrors.Channels;
        var channel_width = args.OutputErrors.Columns;
        var channel_height = args.OutputErrors.Rows;
        const double epsilon = 1e-8;

        // Compute mean and variance across the whole batch per channel
        ComputeMeansAndVariances(args.InputBatch, out var means, out var variances);

        var gamma_gradients = new Matrix<double>[channels];
        var beta_gradients = new Matrix<double>[channels];
        var input_gradients = new Matrix<double>[batches][];
        for (var batch = 0; batch < batches; batch++) {
            input_gradients[batch] = new Matrix<double>[channels];
        }

        // Derivations from https://en.wikipedia.org/wiki/Batch_normalization#:~:text=the%20current%20layer.-,Backpropagation,-%5Bedit%5D
        Parallel.For(0, channels, (int k) => {
            var gamma = Gammas[k];
            var beta = Betas[k];

            // Gradient of L with respect to Beta
            // SUM (dl/dy(k))
            var loss_wrt_b = new Matrix<double>(channel_height, channel_width);
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][k];
                loss_wrt_b.AddWithInplace(loss_wrt_y_k);
            }
            beta_gradients[k] = loss_wrt_b;

            // Gradient of L with respect to Gamma
            // SUM (dl/dy(k) * xHat) where xHat = (y - b) / g
            var loss_wrt_g = new Matrix<double>(channel_height, channel_width);
            for (var batch = 0; batch < batches; batch++) {
                // Compute xHat from y
                // y = g * xHat + b  => xHat = (y - b) / g
                var xhat_k = args.OutputBatch[batch][k] - beta; // Sucks that I have to re-compute this
                xhat_k.ElementWiseInplace(gamma, (xhat, g) => xhat / (g + epsilon));

                var loss_wrt_y_k = args.OutputErrors[batch][k];
                var loss_wrt_y_times_xHat = xhat_k;
                xhat_k.HadamardWithInplace(loss_wrt_y_k);
                loss_wrt_g.AddWithInplace(loss_wrt_y_times_xHat);
            }
            gamma_gradients[k] = loss_wrt_g;

            // Gradient of L with respect to xHat
            var loss_wrt_xhats = new Matrix<double>[batches];
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][k];
                loss_wrt_xhats[batch] = loss_wrt_y_k.HadamardWith(gamma);
            }

            var mean = means[k]; // Mean of the channel over the whole batch
            var variance = variances[k]; // Variance of the channel over the whole batch
            var std = Math.Sqrt(variance + epsilon);
            var scale = 1.0 / std;

            // Gradient of L with respect to variance
            var loss_wrt_variance = new Matrix<double>(channel_height, channel_width);
            var std_three_halves_times_two = -Math.Pow(variance + epsilon, 3.0 / 2.0) * 2.0;
            var one_over_std_three_halves_times_two = 1.0 / std_three_halves_times_two;
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][k];
                var x_k = args.InputBatch[batch][k];

                // Term 1
                var x_minus_mean = x_k.Transform(x => x - mean);
                var x_minus_mean_times_loss = x_minus_mean;
                x_minus_mean.HadamardWithInplace(loss_wrt_y_k);

                // Term 2
                var term1_times_term2 = x_minus_mean_times_loss;
                x_minus_mean_times_loss.ElementWiseInplace(gamma, (lhs, g) => {
                    return lhs * (g * one_over_std_three_halves_times_two);
                });
            }

            // Gradient of L with respect to mean
            var loss_wrt_mean = new Matrix<double>(channel_height, channel_width);
            var loss_wrt_mean_term1 = new Matrix<double>(channel_height, channel_width);
            var loss_wrt_mean_term2 = new Matrix<double>(channel_height, channel_width);
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][k];
                var x_k = args.InputBatch[batch][k];
            
                // Add term 1
                Matrix<double>.ElementWiseInplace(loss_wrt_mean_term1, loss_wrt_y_k, gamma, (y, g) => y * -g * scale);
                loss_wrt_mean.AddWithInplace(loss_wrt_mean_term1);

                // Add term 2
                Matrix<double>.ElementWiseInplace(loss_wrt_mean_term2, loss_wrt_variance, x_k, (v, x) => v * one_over_batches * -2.0 * (x - mean));
                loss_wrt_mean.AddWithInplace(loss_wrt_mean_term2);
            }

            // Gradient of L with respect to x
            for (var batch = 0; batch < batches; batch++) {
                var x = args.InputBatch[batch][k];
                var loss_wrt_xHat = loss_wrt_xhats[batch];
                var loss_wrt_x = new Matrix<double>(x.Rows, x.Columns);

                // Add term 1 (dL/dxHat * scale)
                loss_wrt_x.ElementWiseInplace(loss_wrt_xHat, (_, v) => v * scale);

                // Add term 2 (dl/dV * 2 * (x - mean) / M)
                var temp1 = loss_wrt_variance.ElementWise(x, (v, x) => v * 2 * (x - mean) * one_over_batches);
                loss_wrt_x.AddWithInplace(temp1);

                // Add term 3 (dl/du * 1/M)
                loss_wrt_x.ElementWiseInplace(loss_wrt_mean, (old, u) => old + u * one_over_batches);

                // Save result
                input_gradients[batch][k] = loss_wrt_x;
            }

        });

        return new BackpropagationReturns(
            new BatchedFeatureSet<double>(input_gradients.Select(x => new FeatureSet<double>(x)).ToArray()),
            new Gradients (
                this.Gammas,
                this.Betas,
                gamma_gradients,
                beta_gradients
            )
        );
    }

    public override void SubtractGradients(LayerGradients? gradients) {
        if (gradients is null || gradients is not Gradients grads)
            throw new ArgumentException(nameof(gradients));

        for (var g = 0; g < this.Gammas.Length; g++) {
            this.Gammas[g].SubtractWithInplace(grads.GammaGradients[g]);
        }

        for (var b = 0; b < this.Betas.Length; b++) {
            this.Betas[b].SubtractWithInplace(grads.BetaGradients[b]);
        }
    }

    public override void Initialize(IInitializer initializer) {
        // No need to initialize any weights
    }

    public override int TrainableParameterCount() {
        return InputShape.Count * 2;
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
        private Matrix<double>[] Gammas;
        private Matrix<double>[] Betas;
        public Matrix<double>[] GammaGradients;
        public Matrix<double>[] BetaGradients;

        public Gradients(Matrix<double>[] gamma, Matrix<double>[] beta, Matrix<double>[] gammagrad, Matrix<double>[] betagrad) {
            this.Gammas = gamma;
            this.Betas = beta;
            this.GammaGradients = gammagrad;
            this.BetaGradients = betagrad;
        }

        public override void Clip(double weight_threshold, double bias_threshold) {
            foreach (var matrix in GammaGradients)
                ClipMatrix(matrix, weight_threshold);
            foreach (var matrix in BetaGradients)
                ClipMatrix(matrix, weight_threshold);
        }

        public override void Apply(GradientTransformationHandler handler) {
            int index = 0;
            for (var m = 0; m < GammaGradients.Length; m++) {
                var gradient = GammaGradients[m];
                var parameter = Gammas[m];
                for (var r = 0; r < gradient.Rows; r++) {
                    for (var c = 0; c < gradient.Columns; c++) {
                        gradient[r,c] = handler(index++, parameter[r,c], gradient[r,c]);
                    }
                }
            }

            for (var m = 0; m < BetaGradients.Length; m++) {
                var gradient = BetaGradients[m];
                var parameter = Betas[m];
                for (var r = 0; r < gradient.Rows; r++) {
                    for (var c = 0; c < gradient.Columns; c++) {
                        gradient[r,c] = handler(index++, parameter[r,c], gradient[r,c]);
                    }
                }
            }
        }
    }
}