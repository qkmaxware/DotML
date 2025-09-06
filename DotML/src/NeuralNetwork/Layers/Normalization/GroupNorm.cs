using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer that performs batch normalization. Each channel is put into groups and normalized across the group. 
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
[Untested]
public class GroupNorm : FeedforwardNetworkLayer, INormalizationLayer {
    /// <summary>
    /// Normalization scaling factor
    /// </summary>
    [JsonIgnore] public Matrix<float>[] Gammas {get; set;}
    /// <summary>
    /// Normalization shifting offset
    /// </summary>
    [JsonIgnore] public Matrix<float>[] Betas {get; set;}

    public int NumberOfGroups {get; private set;}

    public GroupNorm (Shape3D input_size, int num_groups) {
        this.NumberOfGroups = num_groups;
        this.InputShape = input_size;
        this.OutputShape = input_size;

        if (input_size.Channels % num_groups != 0) {
            throw new ArgumentException($"Number of groups {num_groups} must divide the number of channels {input_size.Channels} evenly.");
        }

        this.Gammas = new Matrix<float>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Gammas[i] = new Matrix<float>(input_size.Rows, input_size.Columns, 1.0f);
        this.Betas = new Matrix<float>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Betas[i] = new Matrix<float>(input_size.Rows, input_size.Columns, 0.0f);
    }

    private void ComputeMeansAndVariances(FeatureSet<float> features, int range_start, int range_end, out float means, out float variance) {
        if (features.Channels % NumberOfGroups != 0) {
            throw new ArgumentException($"Number of groups {NumberOfGroups} must divide the number of channels {features.Channels} evenly.");
        }

        float mean = 0.0f;
        float m2 = 0.0f;
        int count = 0;
        for (var channelIndex = range_start; channelIndex < range_end; channelIndex++) {
            var matrix = features[channelIndex];
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
        means = mean;
        variance = m2 / count;
    }

    public void ComputeMeansAndVariances(FeatureSet<float> features, out float[] mean_vec, out float[] variance_vec) {
        mean_vec = new float[NumberOfGroups];
        variance_vec = new float[NumberOfGroups];

        var features_per_group = features.Channels / NumberOfGroups;
        for (var group = 0; group < NumberOfGroups; group++) {
            var start_at = group * features_per_group;
            var end_at = start_at + features_per_group;

            ComputeMeansAndVariances(features, start_at, end_at, out var mean, out var variance);
            mean_vec[group] = mean;
            variance_vec[group] = variance;
        }
    }


    const float epsilon = 1e-8f;

    public override FeatureSet<float> EvaluateSync(FeatureSet<float> x) {
        var len = x.Channels;
        Matrix<float>[] outputs = new Matrix<float>[len];

        // Compute the mean and variance across all inputs 
        var features_per_group = x.Channels / NumberOfGroups;
        var groups = this.NumberOfGroups;
        ComputeMeansAndVariances(x, out var means, out var variances);

        for (var group = 0; group < NumberOfGroups; group++) {
            var start_at = group * features_per_group;
            var end_at = start_at + features_per_group;

            var mean = means[group];
            var variance = variances[group];
            var sqrt = 1.0f / MathF.Sqrt(variance + epsilon);      
        
            for (var channel = start_at; channel < end_at; channel++) {
                // Normalize the channel using mean and variance
                var output = x[channel].Transform(v => (v - mean) * sqrt);

                // Apply scaling (gamma) and shifting (beta)
                //output.HadamardWithInplace(Gammas[channel]); // output = output .* gamma
                //output.AddWithInplace(Betas[channel]); // output = output + beta
                output.ElementWiseInplace(Gammas[channel], Betas[channel], (val, gamma, beta) => val * gamma + beta);

                // Save results
                outputs[channel] = output;  
            }
        }

        return new FeatureSet<float>(outputs);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SumAll(Matrix<float>[] values, int start_index, int end_index) {
        float sum = 0.0f;
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
        
        Matrix<float>[] gradient_betas = new Matrix<float>[channels];
        Matrix<float>[] gradient_gammas = new Matrix<float>[channels];
        var input_gradients = new Matrix<float>[batches][];
        for (var batch = 0; batch < batches; batch++) {
            input_gradients[batch] = new Matrix<float>[channels];
        }

        // Compute the mean and variances for for the inputs across each batch
        var mean_per_batch = new float[batches, NumberOfGroups];
        var variance_per_batch = new float[batches, NumberOfGroups];
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
            var gradient_beta = new Matrix<float>(rows, columns);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
                gradient_beta.AddWithInplace((args.OutputErrors[batchIndex])[channel]);
            }
            gradient_betas[channel] = gradient_beta;

            // Gradient of L with respect to gamma
            // dL/dG = Sum_b ( dL/dY * xHat )
            var gradient_gamma = new Matrix<float>(rows, columns);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
                // Compute xHat from y
                Matrix<float> xhat_k;
                xhat_k = args.OutputBatch[batchIndex][channel] - beta;
                xhat_k.ElementWiseInplace(gamma, (xhat, g) => xhat / (g + epsilon));
                
                var loss_wrt_y_k = args.OutputErrors[batchIndex][channel];
                //var loss_wrt_y_times_xHat = xhat_k;
                //loss_wrt_y_times_xHat.HadamardWithInplace(loss_wrt_y_k);
                //gradient_gamma.AddWithInplace(loss_wrt_y_times_xHat);
                gradient_gamma.ElementWiseInplace(xhat_k, loss_wrt_y_k, (val, xk, dyk) => val + xk * dyk);
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
            var loss_wrt_xhats = new Matrix<float>[channels];
            for (var channel = 0; channel < channels; channel++) {
                var gamma = Gammas[channel];
                var loss_wrt_y_k = args.OutputErrors[batch][channel];
                var loss_wrt_xhat = loss_wrt_y_k.HadamardWith(gamma);
                loss_wrt_xhats[channel] = loss_wrt_xhat;
            }

            // Gradient of L with respect to x
            var m = features_per_group * rows * columns;
            var _m = 1.0f / m;
            var var_plus_epsilons = new float[NumberOfGroups];
            var inv_var_plus_epsilons = new float[NumberOfGroups];
            var sqrts = new float[NumberOfGroups];
            var _sqrts = new float[NumberOfGroups];
            var term2_scalars = new float[NumberOfGroups];
            for (var c = 0; c < NumberOfGroups; c++) {
                var var_plus_epsilon = variance_per_batch[batch, c] + epsilon;
                var inv_var_plus_epsilon = 1.0f / var_plus_epsilon;
                var sqrt = MathF.Sqrt(var_plus_epsilon);
                var _sqrt = 1.0f / sqrt;
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

            var sum_all_dxHat_and_x = 0.0f;
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
        private Matrix<float>[] Gammas;
        private Matrix<float>[] Betas;
        public Matrix<float>[] GammaGradients;
        public Matrix<float>[] BetaGradients;

        public Gradients(Matrix<float>[] gamma, Matrix<float>[] beta, Matrix<float>[] gammagrad, Matrix<float>[] betagrad) {
            this.Gammas = gamma;
            this.Betas = beta;
            this.GammaGradients = gammagrad;
            this.BetaGradients = betagrad;
        }

        public override void Clip(float weight_threshold, float bias_threshold) {
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