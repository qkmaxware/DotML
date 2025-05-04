using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

[Untested]
public class GroupNorm : FeedforwardNetworkLayer, INormalizationLayer {
    /// <summary>
    /// Normalization scaling factor
    /// </summary>
    [JsonIgnore] public Matrix<double>[] Gammas {get; set;}
    /// <summary>
    /// Normalization shifting offset
    /// </summary>
    [JsonIgnore] public Matrix<double>[] Betas {get; set;}

    public int NumberOfGroups {get; private set;}

    public GroupNorm (Shape3D input_size, int num_groups) {
        this.NumberOfGroups = num_groups;
        this.InputShape = input_size;
        this.OutputShape = input_size;

        if (input_size.Channels % num_groups != 0) {
            throw new ArgumentException($"Number of groups {num_groups} must divide the number of channels {input_size.Channels} evenly.");
        }

        this.Gammas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Gammas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 1.0);
        this.Betas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Betas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 0.0);
    }

    private void ComputeMeansAndVariances(FeatureSet<double> features, int range_start, int range_end, out double means, out double variance) {
        if (features.Channels % NumberOfGroups != 0) {
            throw new ArgumentException($"Number of groups {NumberOfGroups} must divide the number of channels {features.Channels} evenly.");
        }

        double mean = 0.0;
        double m2 = 0.0;
        int count = 0;
        for (var channelIndex = range_start; channelIndex < range_end; channelIndex++) {
            var matrix = features[channelIndex];
            var rows = matrix.Rows;
            var cols = matrix.Columns;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    count++;
                    double value = matrix[i, j];
                    var delta = value - mean;
                    mean += delta / count;
                    var delta2 = value - mean;
                    m2 += delta * delta2;
                }
            }
        }
        means = mean;
        variance = m2 / count;
    }

    public void ComputeMeansAndVariances(FeatureSet<double> features, out double[] mean_vec, out double[] variance_vec) {
        mean_vec = new double[NumberOfGroups];
        variance_vec = new double[NumberOfGroups];

        var features_per_group = features.Channels / NumberOfGroups;
        for (var group = 0; group < NumberOfGroups; group++) {
            var start_at = group * features_per_group;
            var end_at = start_at + features_per_group;

            ComputeMeansAndVariances(features, start_at, end_at, out var mean, out var variance);
            mean_vec[group] = mean;
            variance_vec[group] = variance;
        }
    }


    const double epsilon = 1e-8;

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> x) {
        var len = x.Channels;
        Matrix<double>[] outputs = new Matrix<double>[len];

        // Compute the mean and variance across all inputs 
        var features_per_group = x.Channels / NumberOfGroups;
        var groups = this.NumberOfGroups;
        ComputeMeansAndVariances(x, out var means, out var variances);

        for (var group = 0; group < NumberOfGroups; group++) {
            var start_at = group * features_per_group;
            var end_at = start_at + features_per_group;

            var mean = means[group];
            var variance = variances[group];
            var sqrt = 1.0 / Math.Sqrt(variance + epsilon);      
        
            for (var channel = start_at; channel < end_at; channel++) {
                // Normalize the channel using mean and variance
                var output = x[channel].Transform(v => (v - mean) * sqrt);

                // Apply scaling (gamma) and shifting (beta)
                output.HadamardWithInplace(Gammas[channel]); // output = output .* gamma
                output.AddWithInplace(Betas[channel]); // output = output + beta

                // Save results
                outputs[channel] = output;  
            }
        }

        return new FeatureSet<double>(outputs);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double SumAll(Matrix<double>[] values, int start_index, int end_index) {
        double sum = 0.0;
        for (var i = start_index; i < end_index; i++) {
            var arr = values[i].AsSpan();
            for (var j = 0; j < arr.Length; j++) {
                sum += arr[j];
            }
        }
        return sum;
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        var (batches, channels, rows, columns) = args.OutputErrors.Shape;
        
        Matrix<double>[] gradient_betas = new Matrix<double>[channels];
        Matrix<double>[] gradient_gammas = new Matrix<double>[channels];
        var input_gradients = new Matrix<double>[batches][];
        for (var batch = 0; batch < batches; batch++) {
            input_gradients[batch] = new Matrix<double>[channels];
        }

        // Compute the mean and variances for for the inputs across each batch
        var mean_per_batch = new double[batches, NumberOfGroups];
        var variance_per_batch = new double[batches, NumberOfGroups];
        for (var b = 0; b < batches; b++) {
            ComputeMeansAndVariances(args.InputBatch[b], out var means, out var variances);
            for (var grp = 0; grp < NumberOfGroups; grp++) {
                mean_per_batch[b, grp] = means[grp];
                variance_per_batch[b, grp] = variances[grp];
            }
        }

        // Compute the gradients of gamma & beta per channel
        var features_per_group = channels / NumberOfGroups;
        for (var channel = 0; channel < channels; channel++) {
            var grp = channel / features_per_group;
            var gamma = Gammas[channel];
            var beta = Betas[channel];

            // Gradient of L with respect to beta
            // dL/dB = Sum_b( dL/dY )
            var gradient_beta = new Matrix<double>(rows, columns);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
                gradient_beta.AddWithInplace((args.OutputErrors[batchIndex])[channel]);
            }
            gradient_betas[channel] = gradient_beta;

            // Gradient of L with respect to gamma
            // dL/dG = Sum_b ( dL/dY * xHat )
            var gradient_gamma = new Matrix<double>(rows, columns);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
                // Compute xHat from y
                Matrix<double> xhat_k;
                xhat_k = args.OutputBatch[batchIndex][channel] - beta;
                xhat_k.ElementWiseInplace(gamma, (xhat, g) => xhat / (g + epsilon));
                
                var loss_wrt_y_k = args.OutputErrors[batchIndex][channel];
                var loss_wrt_y_times_xHat = xhat_k;
                loss_wrt_y_times_xHat.HadamardWithInplace(loss_wrt_y_k);
                gradient_gamma.AddWithInplace(loss_wrt_y_times_xHat);
            }
            gradient_gammas[channel] = gradient_gamma;
        }

        // Compute the gradient of L with respect to x for the input gradients to pass back to the next layers
        for (var batch = 0; batch < batches; batch++) {
            var xs = args.InputBatch[batch];
            var ys = args.OutputBatch[batch];
            var dYs = args.OutputErrors[batch];

            // Gradient of L with respect to xHat
            // dL/dXHat = dL/dY * gamma
            var loss_wrt_xhats = new Matrix<double>[channels];
            for (var channel = 0; channel < channels; channel++) {
                var gamma = Gammas[channel];
                var loss_wrt_y_k = args.OutputErrors[batch][channel];
                var loss_wrt_xhat = loss_wrt_y_k.HadamardWith(gamma);
                loss_wrt_xhats[channel] = loss_wrt_xhat;
            }

            // Gradient of L with respect to x
            var m = features_per_group * rows * columns;
            var _m = 1.0 / m;
            var var_plus_epsilons = new double[NumberOfGroups];
            var inv_var_plus_epsilons = new double[NumberOfGroups];
            var sqrts = new double[NumberOfGroups];
            var _sqrts = new double[NumberOfGroups];
            var term2_scalars = new double[NumberOfGroups];
            for (var c = 0; c < NumberOfGroups; c++) {
                var var_plus_epsilon = variance_per_batch[batch, c] + epsilon;
                var inv_var_plus_epsilon = 1.0 / var_plus_epsilon;
                var sqrt = Math.Sqrt(var_plus_epsilon);
                var _sqrt = 1.0 / sqrt;
                var grp_start = c * features_per_group;
                var grp_end = grp_start + features_per_group;
                var sum_all_dl_dxhat = SumAll(loss_wrt_xhats, grp_start, grp_end);
                var term2_scalar = -sum_all_dl_dxhat * _m * _sqrt;

                var_plus_epsilons[c] = var_plus_epsilon;
                inv_var_plus_epsilons[c] = inv_var_plus_epsilon;
                sqrts[c] = sqrt;
                _sqrts[c] = _sqrt;
                term2_scalars[c] = term2_scalar;
            }

            var sum_all_dxHat_and_x = 0.0;
            {
                var x_count = loss_wrt_xhats.Length;
                for (var i = 0; i < x_count; i++) {
                    // i is channel index
                    var grp = i / features_per_group;
                    var xhat = loss_wrt_xhats[i].AsSpan();
                    var x = xs[i].AsSpan();
                    var el_count = xhat.Length;
                    for (var j = 0; j < el_count; j++) {
                        sum_all_dxHat_and_x += xhat[j] * (x[j] - mean_per_batch[batch, grp]) * inv_var_plus_epsilons[grp];
                    }
                }    
            }
            for (var channel = 0; channel < channels; channel++) {
                var grp = channel / features_per_group;
                var x = xs[channel];
                var y = ys[channel];
                var dY = dYs[channel];
                var dxHat = loss_wrt_xhats[channel];
            
                var term1 = dxHat * _sqrts[grp];

                var term3 = x.Transform((xi) => -(_m * _sqrts[grp]) * (xi - mean_per_batch[batch, grp]) * sum_all_dxHat_and_x);

                input_gradients[batch][channel] = term1.ElementWise(term3, (t1, t3) => t1 + term2_scalars[grp] + t3); // term1 + term2 + term3 | term1,term2 are matrices, term2 is a scalar value 
            }
        }
    
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

    public override void Initialize(IInitializer initializer) { }

    public override int TrainableParameterCount() {
        return InputShape.Count * 2;
    }

    public override void Visit(ILayerVisitor visitor) {
        throw new NotImplementedException();
    }

    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) {
        throw new NotImplementedException();
    }

    public override T Visit<T>(ILayerOutputVisitor<T> visitor) {
        throw new NotImplementedException();
    }

    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) {
        throw new NotImplementedException();
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
}