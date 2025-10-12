using System.Drawing;
using System.Numerics;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer that performs batch normalization. Each channel is normalized across all batches.
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
public class BatchNorm2 : NormalizationLayer
{

    private float running_mean_momentum = 0.9f;
    private Tensor<float> _runningMean;
    public Tensor<float> RunningMean
    {
        get => _runningMean;
        set
        {
            if (!value.Shape.Equals(_runningMean.Shape))
                throw new ArgumentException("Cannot change the shape of the channel running means via assignment");
            _runningMean = value;
        }
    }
    private float running_variance_momentum = 0.9f;
    private Tensor<float> _runningVariance;
    public Tensor<float> RunningVariance
    {
        get => _runningVariance;
        set
        {
            if (!value.Shape.Equals(_runningVariance.Shape))
                throw new ArgumentException("Cannot change the shape of the channel running variances via assignment");
            _runningVariance = value;
        }
    }

    private Tensor<float> _weights;
    public Tensor<float> Weights
    {
        get => _weights;
        set
        {
            if (!value.Shape.Equals(_weights.Shape))
                throw new ArgumentException("Cannot change the shape of the layer weights via assignment");
            _weights = value;
        }
    }
    private Tensor<float> _biases;
    public Tensor<float> Biases
    {
        get => _biases;
        set
        {
            if (!value.Shape.Equals(_biases.Shape))
                throw new ArgumentException("Cannot change the shape of the layer biases via assignment");
            _biases = value;
        }
    }


    public BatchNorm2(int channels)
    {

        _runningMean = Tensor<float>.Ones(new TensorShape(channels));
        _runningVariance = Tensor<float>.Zeros(new TensorShape(channels));

        _weights = Tensor<float>.Ones(new TensorShape(channels));
        _biases = Tensor<float>.Zeros(new TensorShape(channels));
    }

    public override int TrainableParameterCount()
    {
        return Weights.ElementCount + Biases.ElementCount;
    }

    const float epsilon = 1e-8f;

    public override void Initialize(IInitializer initializer)
    {
        var parameters = this.TrainableParameterCount();

        var neurons = Weights.ElementCount;
        Weights.FillGenerated(() => initializer.RandomWeight(neurons, neurons, parameters));
        Biases.FillGenerated(() => initializer.RandomWeight(neurons, neurons, parameters));
    }

    public override TensorShape ForwardShape(TensorShape input) => input;

    public override Tensor<float> Forward(Tensor<float> x) => Forward(x, null); // Inference mode

    public override Tensor<float> Forward(Tensor<float> x, EvaluationContext? ctx)
    {
        var originalRank = x.Shape.Rank;
        var shape = x.Shape.NormalizeRank(4); // Force to be [N, C, H, W]
        x = x.Clone();
        var batches = shape.Length(0); var batchStride = shape.Stride(0);
        var channels = shape.Length(1); var channelStride = shape.Stride(1);
        var arr = x.AsArray();

        for (var channel = 0; channel < channels; channel++)
        {
            // Get references to all spans to this channel across all batches
            var channelInEachBatch = new SpanSurrogate<float>[batches];
            for (var i = 0; i < channelInEachBatch.Length; i++)
            {
                // Can be used to span a SPAN over the array region at a later time. Used because we can't store Spans in an array/span of spans to pass to the MeanAndVariance computation.
                channelInEachBatch[i] = new SpanSurrogate<float>(
                    underlying: arr,
                    offset: i * batchStride + channel * channelStride,
                    count: channelStride
                );
            }

            // Compute the mean and variance across the entire list of channels
            float mean; float variance;
            if (!IsTraining(ctx))
            {
                // During inference always use the computed mean and variances
                mean = this.RunningMean[channel];
                variance = this.RunningVariance[channel];
            }
            else
            {
                // During training always use the computed values out of the batches
                MeanAndVariance(channelInEachBatch, out mean, out variance);
            }
            var sqrt = 1.0f / MathF.Sqrt(variance + epsilon);

            // Update running means and variances which are typically used in Inference or when batch size is too small
            if (batches > 1)
            {
                this.RunningMean[channel] = running_mean_momentum * mean + (1 - running_mean_momentum) * RunningMean[channel];
                this.RunningVariance[channel] = running_variance_momentum * variance + (1 - running_variance_momentum) * RunningVariance[channel];
            }

            var gamma = Weights[channel];
            var beta = Biases[channel];

            foreach (var chan in channelInEachBatch)
            {
                var batch = chan.AsSpan();

                // Normalize
                int i = 0;
                if (Vector.IsHardwareAccelerated && Vector<float>.IsSupported)
                {
                    var sqrtVector = new Vector<float>(sqrt);
                    var meanVector = new Vector<float>(mean);
                    int simdLength = Vector<float>.Count;
                    int simdLimit = batch.Length - simdLength + 1;

                    for (; i < simdLimit; i += simdLength)
                    {
                        var simdSlice = batch.Slice(i, simdLength);
                        var batchVec = new Vector<float>(simdSlice);
                        ((batchVec - meanVector) * sqrtVector).CopyTo(simdSlice);
                    }
                }
                for (; i < batch.Length; i++)
                {
                    batch[i] = (batch[i] - mean) * sqrt;
                }

                // Shift-scale
                i = 0;
                if (Vector.IsHardwareAccelerated && Vector<float>.IsSupported)
                {
                    int simdLength = Vector<float>.Count;
                    int simdLimit = batch.Length - simdLength + 1;
                    var gammaVec = new Vector<float>(gamma);
                    var betaVec = new Vector<float>(beta);

                    for (; i < simdLimit; i += simdLength)
                    {
                        var simdSlice = batch.Slice(i, simdLength);
                        var batchVec = new Vector<float>(simdSlice);
                        ((batchVec * gammaVec) + betaVec).CopyTo(simdSlice);
                    }
                }
                for (; i < batch.Length; i++)
                {
                    batch[i] = batch[i] * gamma + beta;
                }
            }
        }

        var res = x.Squeeze(0..^originalRank);
        
        if (ctx is not null)
        {
            ctx.Save(this, new IOContext(x, res));
        }
        return res;
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        var originalShape = x.Shape;
        var shape = x.Shape.NormalizeRank(4);
        x = x.ReshapeShared(shape);
        y = y.ReshapeShared(shape);
        dy = dy.ReshapeShared(shape);

        int N = shape.Length(0);
        int C = shape.Length(1);
        int H = shape.Length(2);
        int W = shape.Length(3);
        int batchStride = shape.Stride(0);
        int channelStride = shape.Stride(1);
        int spatialSize = H * W;
        int m = N * spatialSize; // total elements per channel

        var gamma = Weights.AsSpan();   // [C]
        var beta = Biases.AsSpan();     // [C]
        var dW_tensor = Tensor<float>.Zeros(Weights.Shape); var dW = dW_tensor.AsSpan();  // dγ
        var dB_tensor = Tensor<float>.Zeros(Biases.Shape); var dB = dB_tensor.AsSpan();   // dβ
        var dX = Tensor<float>.Zeros(x.Shape);
        var dxSpan = dX.AsSpan();

        var xArr = x.AsArray();
        var dyArr = dy.AsArray();
        var xSpan = x.AsSpan();
        var dySpan = dy.AsSpan();

        for (int c = 0; c < C; c++)
        {
            // 1. Gather all [N, H, W] values for this channel across batches
            SpanSurrogate<float>[] xChannel = new SpanSurrogate<float>[N];
            SpanSurrogate<float>[] dyChannel = new SpanSurrogate<float>[N];

            for (int n = 0; n < N; n++)
            {
                int offset = n * batchStride + c * channelStride;
                xChannel[n] = new SpanSurrogate<float>(xArr, offset, channelStride);
                dyChannel[n] = new SpanSurrogate<float>(dyArr, offset, channelStride);
            }

            // 2. Compute mean and variance of x[c] across all batches
            float mean, variance;
            MeanAndVariance(xChannel, out mean, out variance);
            float stdInv = 1.0f / MathF.Sqrt(variance + epsilon);

            float g = gamma[c];

            // 3. First pass: compute x̂ and dy * gamma for accumulation
            float sum_dy = 0f;
            float sum_dy_xhat = 0f;

            // Store temporaries to avoid recomputing
            Span<float> xhat_flat = new float[m];
            Span<float> dy_gamma_flat = new float[m];

            int flatIdx = 0;
            for (int n = 0; n < N; n++)
            {
                var xBatch = xChannel[n].AsSpan();
                var dyBatch = dyChannel[n].AsSpan();

                for (int i = 0; i < xBatch.Length; i++)
                {
                    float xhat = (xBatch[i] - mean) * stdInv;
                    float dy_gamma = dyBatch[i] * g;

                    xhat_flat[flatIdx] = xhat;
                    dy_gamma_flat[flatIdx] = dy_gamma;

                    sum_dy += dy_gamma;
                    sum_dy_xhat += dy_gamma * xhat;

                    // dγ += dy * x̂
                    dW[c] += dyBatch[i] * xhat;

                    // dβ += dy
                    dB[c] += dyBatch[i];

                    flatIdx++;
                }
            }

            float mean1 = sum_dy / m;
            float mean2 = sum_dy_xhat / m;

            // 4. Second pass: compute dX
            flatIdx = 0;
            for (int n = 0; n < N; n++)
            {
                int dxOffset = n * batchStride + c * channelStride;
                var dxBatch = dxSpan.Slice(dxOffset, channelStride);

                for (int i = 0; i < dxBatch.Length; i++)
                {
                    dxBatch[i] = stdInv * (dy_gamma_flat[flatIdx] - mean1 - xhat_flat[flatIdx] * mean2);
                    flatIdx++;
                }
            }
        }

        return new WeightAndBiasGradients(dX.ReshapeShared(originalShape), dW_tensor, dB_tensor);
    }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        if (gradients is not WeightAndBiasGradients wbg)
            throw new ArgumentException("Expected WeightAndBiasGradients", nameof(gradients));

        var dW = wbg.dW;
        var dB = wbg.dB;

        // Apply regularization to weights
        if (regularization is not null)
        {
            dW.ElementWiseBinaryInplace(Weights, (gradient, prevWeight) => gradient + regularization.Invoke(prevWeight));
            dB.ElementWiseBinaryInplace(Biases, (gradient, prevWeight) => gradient + regularization.Invoke(prevWeight));
        }

        // Apply optimizer
        optimizer.UpdateParameter(this, nameof(Weights), learningRate, this.Weights, dW);
        optimizer.UpdateParameter(this, nameof(Biases), learningRate, this.Biases, dB);
    }
    
    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg) => visitor.Visit(this, arg);
}