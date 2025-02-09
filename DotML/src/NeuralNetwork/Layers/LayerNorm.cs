using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer that performs layer (non batch) normalization 
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
[Untested()]
public class LayerNorm : FeedforwardNetworkLayer {

    /// <summary>
    /// Normalization scaling factor
    /// </summary>
    [JsonIgnore] public Matrix<double>[] Gammas {get; set;}
    /// <summary>
    /// Normalization shifting offset
    /// </summary>
    [JsonIgnore] public Matrix<double>[] Betas {get; set;}

    public LayerNorm(Shape3D input_size) {
        this.InputShape = input_size;
        this.OutputShape = input_size;

        this.Gammas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Gammas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 1.0);
        this.Betas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Betas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 0.0);
    }

    public void ComputeMeansAndVariances(FeatureSet<double> features, out double[] mean_vec, out double[] variance_vec) {
        var len = features.Channels;

        var local_means = new double[len];
        var local_variances = new double[len];

        for (var i = 0; i < len; i++) {
            var neurons = features[i];

            var local_mean = neurons.Average();
            var local_variance = neurons.Select(v => Math.Pow(v - local_mean, 2)).Average();

            local_means[i] = local_mean;
            local_variances[i] = local_variance;
        }

        mean_vec = local_means;
        variance_vec = local_variances;
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        var len = channels.Channels;
        Matrix<double>[] outputs = new Matrix<double>[len];

        // Compute the mean and variance across all inputs 
        ComputeMeansAndVariances(channels, out double[] means, out double[] variances);

        // Perform the normalization for each feature
        for (var channel = 0; channel < len; channel++) {
            // Get feature at channel
            Matrix<double> features = channels[channel];

            // Get mean, variance as computed
            var mean = means[channel];
            var variance = variances[channel];
            var sqrt = 1.0 / Math.Sqrt(variance + 1e-8);

            // Normalize the channel using mean and variance
            var output = features.Transform(v => (v - mean) * sqrt);

            // Apply scaling (gamma) and shifting (beta)
            output.HadamardWithInplace(Gammas[channel]); // output = output .* gamma
            output.AddWithInplace(Betas[channel]); // output = output + beta

            // Save results
            outputs[channel] = output;                              
        }

        return new FeatureSet<double>(outputs);
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
        var channels = args.OutputErrors.Channels;
        var rows = args.OutputErrors.Rows;
        var columns = args.OutputErrors.Columns;
        var one_over_features = 1.0 / (rows * columns);
        const double epsilon = 1e-8;

        Matrix<double>[] gradient_betas = new Matrix<double>[channels];
        Matrix<double>[] gradient_gammas = new Matrix<double>[channels];
        var input_gradients = new Matrix<double>[batches][];
        for (var batch = 0; batch < batches; batch++) {
            input_gradients[batch] = new Matrix<double>[channels];
        }

        var means_per_batch = new double[batches][];
        var variances_per_batch = new double[batches][];
        for (var b = 0; b < batches; b++) {
            ComputeMeansAndVariances(args.InputBatch[b], out var means, out var variances);
            means_per_batch[b] = means;
            variances_per_batch[b] = variances;
        }

        Parallel.For(0, channels, channelIndex => {
            var gamma = Gammas[channelIndex];
            var beta = Betas[channelIndex];

            // dL/dB = Sum_b( dL/dY )
            var gradient_beta = new Matrix<double>(rows, columns);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
                gradient_beta.AddWithInplace((args.OutputErrors[batchIndex])[channelIndex]);
            }
            gradient_betas[channelIndex] = gradient_beta;

            // dL/dG = Sum_b ( dL/dY * xHat )
            var gradient_gamma = new Matrix<double>(rows, columns);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
                // Compute xHat from y
                var xhat_k = args.OutputBatch[batchIndex][channelIndex] - beta; // Sucks that I have to re-compute this
                xhat_k.ElementWiseInplace(gamma, (xhat, g) => xhat / (g + epsilon));
                
                var loss_wrt_y_k = args.OutputErrors[batchIndex][channelIndex];
                var loss_wrt_y_times_xHat = xhat_k;
                xhat_k.HadamardWithInplace(loss_wrt_y_k);
                gradient_gamma.AddWithInplace(loss_wrt_y_times_xHat);
            }
            gradient_gammas[channelIndex] = gradient_gamma;

            // Gradient of L with respect to xHat
            var loss_wrt_xhats = new Matrix<double>[batches];
            var avg_loss_xhat = new double[batches];
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][channelIndex];
                var loss_wrt_xhat = loss_wrt_y_k.HadamardWith(gamma);
                loss_wrt_xhats[batch] = loss_wrt_xhat;

                avg_loss_xhat[batch] = loss_wrt_xhat.Average();
            }

            // Gradient of L with respect to x
            for (var batch = 0; batch < batches; batch++) {
                var mean = means_per_batch[batch][channelIndex];
                var variance = variances_per_batch[batch][channelIndex];
                var std = Math.Sqrt(variance + epsilon);
                var scale = 1.0 / std;

                var x = args.InputBatch[batch][channelIndex];
                var loss_wrt_xHat = loss_wrt_xhats[batch];
                var mean_loss_xHat = avg_loss_xhat[batch];
                var loss_wrt_x = new Matrix<double>(x.Rows, x.Columns);

                // Once we have the gradients with respect to the mean and variance, we can compute the gradient with respect to the input tensor `x`: 
                /*
                \[
                \frac{\partial L}{\partial x_{b,c,h,w}} = \frac{1}{\sqrt{\sigma_c^2 + \epsilon}} \left( \frac{\partial L_{b,c,h,w}}{\partial \hat{x}_{b,c,h,w}} - \frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} \frac{\partial L_{b,c,h,w}}{\partial \hat{x}_{b,c,h,w}} \right) - \frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} \frac{\partial L_{b,c,h,w}}{\partial \hat{x}_{b,c,h,w}} \cdot (x_{b,c,h,w} - \mu_c)
                \]
                */

                // Add term 1 (1/sqrt(variance + e) * (dL/dxHat - mean(dL/dxHat)))
                loss_wrt_x.ElementWiseInplace(loss_wrt_xHat, (_, xHat) => scale * (xHat - mean_loss_xHat));

                // Add term 2 (mean(dL/dxHat) * (x - mean))
                loss_wrt_x.ElementWiseInplace(x, (old, x) => old - mean_loss_xHat * (x - mean));

                // Save result
                input_gradients[batch][channelIndex] = loss_wrt_x;
            }
        });

        return new BackpropagationReturns(
            new BatchedFeatureSet<double>(input_gradients.Select(x => new FeatureSet<double>(x)).ToArray()),
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

        for (var g = 0; g < this.Gammas.Length; g++) {
            this.Gammas[g].SubtractWithInplace(grads.GammaGradients[g]);
        }

        for (var b = 0; b < this.Betas.Length; b++) {
            this.Betas[b].SubtractWithInplace(grads.BetaGradients[b]);
        }
    }

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

    public override void Initialize(IInitializer initializer) { }

    public override int TrainableParameterCount() {
        return InputShape.Count * 2;
    }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(ILayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}