using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer that performs layer (non batch) normalization. Each channel is normalized across all channels.
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
public class LayerNorm : FeedforwardNetworkLayer, INormalizationLayer {

    /// <summary>
    /// Normalization scaling factor
    /// </summary>
    [JsonIgnore] public Matrix<float>[] Gammas {get; set;}
    /// <summary>
    /// Normalization shifting offset
    /// </summary>
    [JsonIgnore] public Matrix<float>[] Betas {get; set;}

    public LayerNorm(Shape3D input_size) {
        this.InputShape = input_size;
        this.OutputShape = input_size;

        this.Gammas = new Matrix<float>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Gammas[i] = new Matrix<float>(input_size.Rows, input_size.Columns, 1.0f);
        this.Betas = new Matrix<float>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Betas[i] = new Matrix<float>(input_size.Rows, input_size.Columns, 0.0f);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ComputeMeansAndVariances(FeatureSet<float> features, out float mean_vec, out float variance_vec) {
        if (
            Vector<float>.IsSupported
            && Vector.IsHardwareAccelerated
        ) {
            ComputeMeansAndVariancesVector(features, out mean_vec, out variance_vec);
        } else {
            ComputeMeansAndVariancesArray(features, out mean_vec, out variance_vec);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ComputeMeansAndVariancesVector(FeatureSet<float> features, out float mean_vec, out float variance_vec) {
        var channels = features.Channels;
        var item_count = channels * features.Rows * features.Columns;

        var mean_sum = 0.0f;
        var vec_sum = Vector<float>.Zero;
        var vec_size = Vector<float>.Count;
        for (var c = 0; c < channels; c++) {
            var arr = features[c].AsSpan();
            int i = 0;
            
            for (; i < arr.Length - vec_size; i += vec_size) {
                vec_sum += new Vector<float>(arr.Slice(i, vec_size));
            }

            for (; i < arr.Length; i++) {
                mean_sum += arr[i];
            }
        }
        mean_sum += Vector.Sum(vec_sum);
        var layer_mean = mean_sum / item_count;

        var variance_sum = 0.0f;
        vec_sum = Vector<float>.Zero;
        var layer_mean_vec = new Vector<float>(layer_mean);
        for (var c = 0; c < channels; c++) {
            var arr = features[c].AsSpan();
            int i = 0;

            for (; i < arr.Length - vec_size; i += vec_size) {
                var ds = Vector.Subtract(new Vector<float>(arr.Slice(i, vec_size)), layer_mean_vec);
                vec_sum += Vector.Multiply(ds, ds);
            }

            for (; i < arr.Length; i++) {
                var d = arr[i] - layer_mean;
                variance_sum += d * d;
            }
        }
        variance_sum += Vector.Sum(vec_sum);

        mean_vec = layer_mean;
        variance_vec = variance_sum / item_count;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeMeansAndVariancesArray(FeatureSet<float> features, out float mean_vec, out float variance_vec) {
        var channels = features.Channels;
        var item_count = channels * features.Rows * features.Columns;

        var mean_sum = 0.0f;
        for (var c = 0; c < channels; c++) {
            var arr = features[c].AsSpan();
            int i = 0;

            for (; i < arr.Length - 4; i += 4) {
                mean_sum += arr[i] + arr[i+1] + arr[i+2] + arr[i+3];
            }

            for (; i < arr.Length; i++) {
                mean_sum += arr[i];
            }
        }
        var layer_mean = mean_sum / item_count;

        var variance_sum = 0.0f;
        for (var c = 0; c < channels; c++) {
            var arr = features[c].AsSpan();
            int i = 0;

            for (; i < arr.Length - 4; i += 4) {
                var d0 = arr[i]   - layer_mean;
                var d1 = arr[i+1] - layer_mean;
                var d2 = arr[i+2] - layer_mean;
                var d3 = arr[i+3] - layer_mean;

                variance_sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            }

            for (; i < arr.Length; i++) {
                var d = arr[i] - layer_mean;
                variance_sum += d * d;
            }
        }

        mean_vec = layer_mean;
        variance_vec = variance_sum / item_count;
    }

    const float epsilon = 1e-8f;

    public override FeatureSet<float> EvaluateSync(FeatureSet<float> channels) {
        var len = channels.Channels;
        //Matrix<double>[] norms = new Matrix<double>[len];
        Matrix<float>[] outputs = new Matrix<float>[len];

        // Compute the mean and variance across all inputs 
        ComputeMeansAndVariances(channels, out float mean, out float variance);
        var sqrt = 1.0f / MathF.Sqrt(variance + epsilon);

        // Perform the normalization for each feature
        for (var channel = 0; channel < len; channel++) {
            // Get feature at channel
            Matrix<float> features = channels[channel];

            // Normalize the channel using mean and variance
            var output = features.Transform(v => (v - mean) * sqrt);
            // norms[channel] = norm; // Save normalized feature

            // Apply scaling (gamma) and shifting (beta)
            // output = output.HadamardWith(Gammas[channel]);
            output.ElementWiseInplace(Gammas[channel], Betas[channel], (val, gamma, beta) => val * gamma + beta);
            //output.HadamardWithInplace(Gammas[channel]); // output = output .* gamma
            //output.AddWithInplace(Betas[channel]); // output = output + beta

            // Save results
            outputs[channel] = output;                              
        }

        return new FeatureSet<float>(outputs);
        //return new LayerNormFeatureSet(channels, mean, variance, new FeatureSet<double>(norms), outputs);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SumAll(Matrix<float>[] values) {
        float sum = 0.0f;
        for (var i = 0; i < values.Length; i++) {
            var arr = values[i].AsSpan();
            for (var j = 0; j < arr.Length; j++) {
                sum += arr[j];
            }
        }
        return sum;
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        var batches  = args.OutputErrors.Batches;
        var channels = args.OutputErrors.Channels;
        var rows = args.OutputErrors.Rows;
        var columns = args.OutputErrors.Columns;
        var one_over_features = 1.0f / (rows * columns);
    
        Matrix<float>[] gradient_betas = new Matrix<float>[channels];
        Matrix<float>[] gradient_gammas = new Matrix<float>[channels];
        var input_gradients = new Matrix<float>[batches][];
        for (var batch = 0; batch < batches; batch++) {
            input_gradients[batch] = new Matrix<float>[channels];
        }

        // Compute the mean and variances for for the inputs across each batch
        var mean_per_batch = new float[batches];
        var variance_per_batch = new float[batches];
        for (var b = 0; b < batches; b++) {
            //if (args.OutputBatch[b] is LayerNormFeatureSet feats) {
                // Fetch it
                //mean_per_batch[b] = feats.Mean;
                //variance_per_batch[b] = feats.Variance;
                //continue;
            //} else {
                // Recompute it
                ComputeMeansAndVariances(args.InputBatch[b], out var means, out var variances);
                mean_per_batch[b] = means;
                variance_per_batch[b] = variances;
            //}
        }

        // Compute the gradients of gamma & beta per channel
        for (var channel = 0; channel < channels; channel++) {
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
                //if (args.OutputBatch[batchIndex] is LayerNormFeatureSet feats) {
                    // Fetch it
                    //xhat_k = feats.NormalizedInput[channel].Clone();
                //} else {
                    // Recompute it
                    xhat_k = args.OutputBatch[batchIndex][channel] - beta;
                    xhat_k.ElementWiseInplace(gamma, (xhat, g) => xhat / (g + epsilon));
                //}
                
                var loss_wrt_y_k = args.OutputErrors[batchIndex][channel];
                //var loss_wrt_y_times_xHat = xhat_k;
                //loss_wrt_y_times_xHat.HadamardWithInplace(loss_wrt_y_k);  
                //gradient_gamma.AddWithInplace(loss_wrt_y_times_xHat); //gradient_gamma .+ xhat_k .* loss_wrt_y_k
                gradient_gamma.ElementWiseInplace(xhat_k, loss_wrt_y_k, (val, xk, dyk) =>  val + xk * dyk);
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
            var var = variance_per_batch[batch];
            var mean = mean_per_batch[batch];
            var m = InputShape.Count; // channels * x.Rows * x.Columns;
            var _m = 1.0f / m;
            var var_plus_epsilon = var + epsilon;
            var inv_var_plus_epsilon = 1.0f / var_plus_epsilon;
            var sqrt = MathF.Sqrt(var_plus_epsilon);
            var _sqrt = 1.0f / sqrt;
            var sum_all_dl_dxhat = SumAll(loss_wrt_xhats); //loss_wrt_xhats.SelectMany(xhat => xhat).Sum();
            var term2_scalar = -sum_all_dl_dxhat / (m * sqrt);

            var sum_all_dxHat_and_x = 0.0f;
            {
                var x_count = loss_wrt_xhats.Length;
                for (var i = 0; i < x_count; i++) {
                    var xhat = loss_wrt_xhats[i].AsSpan();
                    var x = xs[i].AsSpan();
                    var el_count = xhat.Length;
                    for (var j = 0; j < el_count; j++) {
                        sum_all_dxHat_and_x += xhat[j] * (x[j] - mean) * inv_var_plus_epsilon;
                    }
                }    
            }
            /*var sum_all_dxHat_and_x = loss_wrt_xhats.Zip(args.InputBatch[batch]).SelectMany((xhat_x_pair) => xhat_x_pair.First.Zip(xhat_x_pair.Second)).Select((pair) => {
                var (xhat, x) = pair;
                return xhat * (x - mean) * inv_var_plus_epsilon;
            }).Sum();*/
            for (var channel = 0; channel < channels; channel++) {
                var x = xs[channel];
                var y = ys[channel];
                var dY = dYs[channel];
                var dxHat = loss_wrt_xhats[channel];
            
                var term1 = dxHat * _sqrt;

                var term3 = x.Transform((xi) => -(_m * _sqrt) * (xi - mean) * sum_all_dxHat_and_x);

                input_gradients[batch][channel] = term1.ElementWise(term3, (t1, t3) => t1 + term2_scalar + t3); // term1 + term2 + term3 | term1,term2 are matrices, term2 is a scalar value 
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

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => visitor.Visit(this, args);
    public override T Visit<T>(ILayerOutputVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);

    /*private class LayerNormFeatureSet : FeatureSet<double> {
        public FeatureSet<double> Input {get; set;}
        public FeatureSet<double> NormalizedInput {get; set;}
        public double Mean {get; set;}
        public double Variance {get; set;}
        public LayerNormFeatureSet(FeatureSet<double> x, double mean, double variance, FeatureSet<double> normalized_x, Matrix<double>[] features) : base(features) {
            this.Input = x;
            this.NormalizedInput = normalized_x;
            this.Mean = mean;
            this.Variance = variance;
        }
    }*/

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