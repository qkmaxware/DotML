using System.Runtime.CompilerServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public class TransposeConv2D : NetworkLayer
{
    public int Groups { get; set; }
    public (int X, int Y) Stride { get; set; }
    public (int X, int Y) Dilation { get; set; }
    public (int Left, int Top, int Right, int Bottom) InputPadding { get; set; }
    public (int Left, int Top, int Right, int Bottom) OutputPadding { get; set; }
    public Tensor<float> Weights;   // [outChannels, inChannelsPerGroup, kernelHeight, kernelWidth]
    public Tensor<float> Biases;    // [outChannels]

    public TransposeConv2D(int outChannels, int inChannelsPerGroup, int groups, (int Width, int Height) kernel, (int X, int Y) stride, (int X, int Y) dilation, (int Left, int Top, int Right, int Bottom) inputPadding, (int Left, int Top, int Right, int Bottom) outputPadding)
    {
        this.Weights = Tensor<float>.Ones(new TensorShape(outChannels, inChannelsPerGroup, kernel.Height, kernel.Width));
        this.Biases = Tensor<float>.Zeros(new TensorShape(outChannels));

        this.Groups = groups;
        this.Stride = stride;
        this.Dilation = dilation;
        this.InputPadding = inputPadding;
        this.OutputPadding = outputPadding;
    }

    public override void Initialize(IInitializer initializer)
    {
        var parameters = this.TrainableParameterCount();
        Weights.FillGenerated(() => initializer.RandomWeight(parameters, parameters, parameters));
        Biases.FillGenerated(() => initializer.RandomBias(parameters, parameters, parameters));
    }

    public override TensorShape ForwardShape(TensorShape input)
    {
        // See Tensor<T>.TransposeConvolve2D
        // Normalize all tensors to 4D (expand or reduce as required)
        input = input.NormalizeRank(4);// [batch, channels, rows, columns]
        var kernels = this.Weights.Shape.NormalizeRank(4);        // [inChannelsGrouped, outChannelsPerGroup, kernelHeight, kernelWidth]

        var batch = input.Length(0);
        var inChannels = input.Length(1);
        var inHeight = input.Length(2);
        var inWidth = input.Length(3);

        var inChannelsPerGroup = inChannels / this.Groups;
        var outChannelsPerGroup = kernels.Length(1);
        var outChannels = outChannelsPerGroup * this.Groups;

        if (inChannels % this.Groups != 0)
            throw new ArgumentException("Input channels must be divisible by the number of groups.");
        if (kernels.Length(0) != inChannelsPerGroup)
            throw new ArgumentException("Kernel input channels do not match expected channels per group.");
        if (kernels.Length(1) * this.Groups != outChannels)
            throw new ArgumentException("Kernel output channels do not match expected channels per group.");

        var kernelHeight = kernels.Length(2);
        var kernelWidth = kernels.Length(3);

        // Compute output size (based on standard transposed conv formula)
        var outHeight = (inHeight - 1) * Stride.Y - InputPadding.Top - InputPadding.Bottom + Dilation.Y * (kernelHeight - 1) + 1 + OutputPadding.Top + OutputPadding.Bottom;
        var outWidth = (inWidth - 1) * Stride.X - InputPadding.Left - InputPadding.Right + Dilation.X * (kernelWidth - 1) + 1 + OutputPadding.Left + OutputPadding.Right;

        var outputShape = new TensorShape(batch, outChannels, outHeight, outWidth);
        return outputShape;
    }

    public override Tensor<float> Forward(Tensor<float> channels)
    {
        return channels.TransposeConvolve2D(
            kernels: this.Weights,
            groups: this.Groups,
            strideX: this.Stride.X,
            strideY: this.Stride.Y,
            dilationX: this.Dilation.X,
            dilationY: this.Dilation.Y,
            inPadLeft: this.InputPadding.Left,
            inPadRight: this.InputPadding.Right,
            inPadTop: this.InputPadding.Top,
            inPadBottom: this.InputPadding.Bottom,
            outPadLeft: this.OutputPadding.Left,
            outPadRight: this.OutputPadding.Right,
            outPadTop: this.OutputPadding.Top,
            outPadBottom: this.OutputPadding.Bottom,
            bias: this.Biases.AsSpan() // Per channel bias
        );
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        // Shapes
        var xShape = x.Shape.NormalizeRank(4);                     // [batch, channels, rows, columns]

        var batch = xShape.Length(0);
        var inChannels = xShape.Length(1);
        var inHeight = xShape.Length(2);
        var inWidth = xShape.Length(3);

        var outChannels = dy.Shape.Length(1);
        var outHeight = dy.Shape.Length(2);
        var outWidth = dy.Shape.Length(3);

        var kernelHeight = Weights.Shape.Length(2);
        var kernelWidth = Weights.Shape.Length(3);

        var inChannelsPerGroup = inChannels / Groups;
        var outChannelsPerGroup = outChannels / Groups;

        var flippedWeights = Weights.Mirror(^2, ^1); // Flip kernelHeight and kernelWidth
        var swappedWeights = flippedWeights.Permute(1, 0, 2, 3); // Swap in/out channels

        // Gradient w.r.t input - Convolve the output gradient with the weights flipped spatially
        var dX = dy.Convolve2D(
            kernels: swappedWeights,
            groups: Groups,
            strideX: Stride.X,
            strideY: Stride.Y,
            dilationX: Dilation.X,
            dilationY: Dilation.Y,
            padLeft: OutputPadding.Left,
            padRight: OutputPadding.Right,
            padTop: OutputPadding.Top,
            padBottom: OutputPadding.Bottom
        );

        // Gradient w.r.t weights - Convolve the input with the output gradient summing over the batch
        var dW = Tensor<float>.Zeros(Weights.Shape); // [outChannels, inChannelsPerGroup, kernelHeight, kernelWidth]
        for (int g = 0; g < Groups; g++)
        {
            int inOffset = g * inChannelsPerGroup;
            int outOffset = g * outChannelsPerGroup;

            for (int oc = 0; oc < outChannelsPerGroup; oc++)
            {
                for (int ic = 0; ic < inChannelsPerGroup; ic++)
                {
                    for (int kh = 0; kh < kernelHeight; kh++)
                    {
                        for (int kw = 0; kw < kernelWidth; kw++)
                        {
                            float grad = 0f;
                            for (int b = 0; b < batch; b++)
                            {
                                for (int ih = 0; ih < inHeight; ih++)
                                {
                                    for (int iw = 0; iw < inWidth; iw++)
                                    {
                                        // Calculate output position for this input pixel and kernel offset
                                        int oh = ih * Stride.Y - InputPadding.Top + kh * Dilation.Y + OutputPadding.Top;
                                        int ow = iw * Stride.X - InputPadding.Left + kw * Dilation.X + OutputPadding.Left;

                                        if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth)
                                        {
                                            float xval = x[b, inOffset + ic, ih, iw];
                                            float dyval = dy[b, outOffset + oc, oh, ow];
                                            grad += xval * dyval;
                                        }
                                    }
                                }
                            }
                            dW[outOffset + oc, ic, kh, kw] = grad;
                        }
                    }
                }
            }
        }

        // Gradient w.r.t. biases - just sum over N, H_out, W_out
        var dB = dy.Sum(axes: [0, 2, 3], keepdim: false);

        return new WeightAndBiasGradients(dX, dW, dB);
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