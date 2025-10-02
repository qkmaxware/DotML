using System.Runtime.CompilerServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Apply a convolution using the given kernel/filter
/// <see href="https://en.wikipedia.org/wiki/Convolutional_layer"/>
/// </summary>
public class Conv2D : NetworkLayer
{

    public int Groups { get; set; }
    public (int X, int Y) Stride { get; set; }
    public (int X, int Y) Dilation { get; set; }
    public (int Left, int Top, int Right, int Bottom) Padding { get; set; }
    public Tensor<float> Weights;   // [outChannels, inChannelsPerGroup, kernelHeight, kernelWidth]
    public Tensor<float> Biases;    // [outChannels]

    public Conv2D(int outChannels, int inChannelsPerGroup, int groups, (int Width, int Height) kernel, (int X, int Y) stride, (int X, int Y) dilation, (int Left, int Top, int Right, int Bottom) padding)
    {
        this.Weights = Tensor<float>.Ones(new TensorShape(outChannels, inChannelsPerGroup, kernel.Height, kernel.Width));
        this.Biases = Tensor<float>.Zeros(new TensorShape(outChannels));

        this.Groups = groups;
        this.Stride = stride;
        this.Dilation = dilation;
        this.Padding = padding;
    }

    public override int TrainableParameterCount()
    {
        return Weights.ElementCount + Biases.ElementCount;
    }

    public override void Initialize(IInitializer initializer)
    {
        var parameters = this.TrainableParameterCount();
        Weights.FillGenerated(() => initializer.RandomWeight(parameters, parameters, parameters));
        Biases.FillGenerated(() => initializer.RandomBias(parameters, parameters, parameters));
    }

    public override TensorShape ForwardShape(TensorShape input)
    {
        // See Tensor.Convolve2D
        input = input.NormalizeRank(4);                     // [batch, channels, rows, columns]
        var kernels = Weights.Shape.NormalizeRank(4);       // [outChannels, inChannelsPerGroup, kernelHeight, kernelWidth]

        var batch = input.Length(0);
        var inChannels = input.Length(1);
        var inHeight = input.Length(2);
        var inWidth = input.Length(3);

        var outChannels = kernels.Length(0);
        var inChannelsPerGroup = inChannels / Groups;

        if (inChannels % Groups != 0)
            throw new ArgumentException("Input channels must be divisible by the number of groups.");
        if (outChannels % Groups != 0)
            throw new ArgumentException("Output channels must be divisible by the number of groups.");
        if (kernels.Length(1) != inChannelsPerGroup)
            throw new ArgumentException("Kernel input channels do not match expected channels per group.");

        var kernelHeight = kernels.Length(2);
        var kernelWidth = kernels.Length(3);

        var outHeight = (inHeight + Padding.Top + Padding.Bottom - Dilation.Y * (kernelHeight - 1) - 1) / Stride.Y + 1;
        var outWidth = (inWidth + Padding.Left + Padding.Right - Dilation.X * (kernelWidth - 1) - 1) / Stride.X + 1;
        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException("Invalid output dimensions. Check padding, stride, and dilation.");

        return new TensorShape(batch, outChannels, outHeight, outWidth);
    }

    public override Tensor<float> Forward(Tensor<float> channels)
    {
        return channels.Convolve2D(
            kernels: this.Weights,
            groups: this.Groups,
            strideX: this.Stride.X,
            strideY: this.Stride.Y,
            dilationX: this.Dilation.X,
            dilationY: this.Dilation.Y,
            padLeft: this.Padding.Left,
            padRight: this.Padding.Right,
            padTop: this.Padding.Top,
            padBottom: this.Padding.Bottom,
            bias: this.Biases.AsSpan() // Per channel bias
        );
    }


    public override Gradients Backward(Tensor<float> x, Tensor<float> _y, Tensor<float> dy)
    {
        // Gradient w.r.t. biases: just sum over N, H_out, W_out
        Tensor<float> dB = dy.Sum(axes: [0, 2, 3], keepdim: false); // shape: [C_out]

        // Gradient w.r.t. weights
        Tensor<float> dW = Tensor<float>.Zeros(this.Weights.Shape);
        int N = x.Shape.Length(0);
        int C_out = dy.Shape.Length(1);
        int C_in = x.Shape.Length(1);
        int H_in = x.Shape.Length(2);
        int W_in = x.Shape.Length(3);
        int H_out = dy.Shape.Length(2);
        int W_out = dy.Shape.Length(3);
        int H_k = this.Weights.Shape.Length(2);
        int W_k = this.Weights.Shape.Length(3);

        int G = this.Groups;
        int C_in_per_group = C_in / G;
        int C_out_per_group = C_out / G;

        for (int n = 0; n < N; n++)
        {
            for (int g = 0; g < G; g++)
            {
                for (int oc = 0; oc < C_out_per_group; oc++)
                {
                    int outChannel = g * C_out_per_group + oc;

                    for (int ic = 0; ic < C_in_per_group; ic++)
                    {
                        int inChannel = g * C_in_per_group + ic;

                        for (int kh = 0; kh < H_k; kh++)
                        {
                            for (int kw = 0; kw < W_k; kw++)
                            {
                                float sum = 0f;

                                for (int y = 0; y < H_out; y++)
                                {
                                    int in_y = y * Stride.Y - Padding.Top + kh * Dilation.Y;
                                    if (in_y < 0 || in_y >= H_in) continue;

                                    for (int x_ = 0; x_ < W_out; x_++)
                                    {
                                        int in_x = x_ * Stride.X - Padding.Left + kw * Dilation.X;
                                        if (in_x < 0 || in_x >= W_in) continue;

                                        float inputVal = x[n, inChannel, in_y, in_x];
                                        float gradOutVal = dy[n, outChannel, y, x_];

                                        sum += inputVal * gradOutVal;
                                    }
                                }

                                dW[outChannel, ic, kh, kw] += sum;
                            }
                        }
                    }
                }
            }
        }

        // Gradient w.r.t. input
        Tensor<float> dx = dy.TransposeConvolve2D(
            kernels: this.Weights,
            groups: this.Groups,
            strideX: this.Stride.X,
            strideY: this.Stride.Y,
            dilationX: this.Dilation.X,
            dilationY: this.Dilation.Y,
            inCropLeft: this.Padding.Left,       // Cropping
            inCropRight: this.Padding.Right,     // Cropping
            inCropTop: this.Padding.Top,         // Cropping
            inCropBottom: this.Padding.Bottom,   // Cropping
            outPadLeft: 0,
            outPadRight: 0,
            outPadTop: 0,
            outPadBottom: 0
        );

        return new WeightAndBiasGradients(
            dx: dx,
            dw: dW,
            db: dB
        );
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