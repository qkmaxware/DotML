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
    public Tensor<float> Weights;   // [inChannelsPerGroup, outChannels, kernelHeight, kernelWidth]
    public Tensor<float> Biases;    // [outChannels]

    public TransposeConv2D(int outChannels, int inChannelsPerGroup, int groups, (int Width, int Height) kernel, (int X, int Y) stride, (int X, int Y) dilation, (int Left, int Top, int Right, int Bottom) inputPadding, (int Left, int Top, int Right, int Bottom) outputPadding)
    {
        this.Weights = Tensor<float>.Ones(new TensorShape(inChannelsPerGroup, outChannels, kernel.Height, kernel.Width));
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

        // Gradient w.r.t. biases - just sum over N, H_out, W_out
        var dB = dy.Sum(axes: [0, 2, 3], keepdim: false);

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
                            dW[ic, outOffset + oc, kh, kw] = grad;
                        }
                    }
                }
            }
        }

        // Gradient w.r.t input - Convolve the output gradient with the weights flipped spatially and swapped input/output channels
        /*var weightsForDX = Weights
            .ReshapeShared(new TensorShape(
                inChannelsPerGroup, // C_in
                Groups,
                outChannelsPerGroup,
                kernelHeight,
                kernelWidth
            ))
            .Permute(1, 2, 0, 3, 4) // [Groups, outPerGroup, inPerGroup, H, W]
            .ReshapeShared(new TensorShape(
                inChannels, // Groups * inPerGroup
                outChannelsPerGroup,
                kernelHeight,
                kernelWidth
            ))
            .Mirror(^2, ^1); // Flip H, W
        ;

        // Compute padding for backward pass (see transposed conv backward math)
        inHeight = xShape.Length(2) + InputPadding.Top + InputPadding.Bottom;
        inWidth = xShape.Length(3) + InputPadding.Left + InputPadding.Right;

        int totalPadX = (inWidth - 1) * Stride.X + Dilation.X * (kernelWidth - 1) + 1 - outWidth;
        int padLeft = totalPadX / 2;
        int padRight = totalPadX - padLeft;

        int totalPadY = (inHeight - 1) * Stride.Y + Dilation.Y * (kernelHeight - 1) + 1 - outHeight;
        int padTop = totalPadY / 2;
        int padBottom = totalPadY - padTop;

        
        var dX = dy.Convolve2D(
            kernels: weightsForDX,
            groups: Groups,
            strideX: 1,
            strideY: 1,
            dilationX: Stride.X,
            dilationY: Stride.Y,
            padLeft: padLeft,
            padRight: padRight,
            padTop: padTop,
            padBottom: padBottom
        );*/
        var dX = BackpropagateWrtInput(xShape, dy);

        return new WeightAndBiasGradients(dX.ReshapeShared(x.Shape), dW, dB);
    }

    private Tensor<float> BackpropagateWrtInput(TensorShape xShape, Tensor<float> dY) {
        var (batch_size, in_channels, out_rows, out_columns) = xShape; // In and out rows/columns flipped here since the "input" is dY and the output is "dX" for the convolution
        var (inChannelsPerGroup, out_channels, kernelHeight, kernelWidth) = Weights.Shape;
        var (_, _, in_rows, in_columns) = dY.Shape;

        var dyStride0 = dY.Shape.Stride(0);
        var dyStride1 = dY.Shape.Stride(1);

        var wStride0 = Weights.Shape.Stride(0);
        var wStride1 = Weights.Shape.Stride(1);

        var dX = Tensor<float>.Zeros(xShape);
        var dxStride0 = dX.Shape.Stride(0);
        var dxStride1 = dX.Shape.Stride(1);
        
        int inChannelsPerGroupCount = inChannelsPerGroup; // from Weights.Shape
        int outChannelsPerGroupCount = out_channels / Groups;

        // How it worked.
        // Each filter was an output channel
        // Each kernel applied to a single input
        // So to go backwards we need to take the output from each filter and distribute it with each kernel back to the associated input
        for (var batch = 0; batch < batch_size; batch++)
        {

            for (var group = 0; group < Groups; group++)
            {
                int inChannelStart = group * inChannelsPerGroupCount;
                int inChannelEnd = inChannelStart + inChannelsPerGroupCount;

                int outChannelStart = group * outChannelsPerGroupCount;
                int outChannelEnd = outChannelStart + outChannelsPerGroupCount;

                for (var output_index = 0; output_index < out_channels; output_index++)
                {
                    int localOutputIndex = output_index - outChannelStart;
                    
                    ReadOnlySpan2D<float> ouput_span = new ReadOnlySpan2D<float>(dY.AsSpan(batch * dyStride0 + output_index * dyStride1, dyStride1), in_rows, in_columns); // TODO ergonomic tensor method for this?

                    for (var kernel_index = 0; kernel_index < in_channels; kernel_index++)
                    {
                        int localInputIndex = kernel_index - inChannelStart;

                        ReadOnlySpan2D<float> kernel_span = new ReadOnlySpan2D<float>(Weights.AsSpan(localInputIndex * wStride0 + localOutputIndex * wStride1, wStride1), kernelHeight, kernelWidth); // TODO ergonomic tensor method for this?
                        Span2D<float> result_span = new Span2D<float>(dX.AsSpan(batch * dxStride0 + kernel_index * dxStride1, dxStride1), out_rows, out_columns); // TODO ergonomic tensor method for this?

                        // --------------------------------------
                        // COPIED FROM Matrix<double>.Convolve();
                        // --------------------------------------
                        for (var y = 0; y < out_rows; y++)
                        {
                            var startY = y * Stride.Y - InputPadding.Top;

                            for (var x = 0; x < out_columns; x++)
                            {
                                var startX = x * Stride.X - InputPadding.Left;

                                var total_sum = 0.0f;
                                for (int ky = 0; ky < kernelHeight; ky++)
                                {
                                    var inY = startY + ky;
                                    if (inY < 0 || inY >= in_rows) continue; // Skip out-of-bounds rows

                                    for (int kx = 0; kx < kernelWidth; kx++)
                                    {
                                        var inX = startX + kx;
                                        if (inX < 0 || inX >= in_columns) continue; // Skip out-of-bounds columns

                                        //var kernel = Weights[localInputIndex, localOutputIndex, ky, kx];
                                        var kernel = kernel_span[ky, kx];
                                        //var output = dY[batch, output_index, inY, inX];
                                        var output = ouput_span[inY, inX];
                                        total_sum += output * kernel;
                                    }
                                }

                                //dX[batch, kernel_index, y, x] += total_sum;
                                result_span[y, x] += total_sum;
                            }
                        }
                        // --------------------------------------
                    }
                }
            }
        }

        return dX;
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