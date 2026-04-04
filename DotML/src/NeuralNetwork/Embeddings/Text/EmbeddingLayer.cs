using System.Security.Cryptography;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network.Embedding.Text;
/*
public class EmbeddingGradients : Gradients
{
    public Gradients TokenGradients { get; }
    public Gradients PositionalGradients { get; }

    public EmbeddingGradients(
        Tensor<float> dX,
        Gradients tokenGradients,
        Gradients positionalGradients
        ): base(dX)
    {
        TokenGradients = tokenGradients;
        PositionalGradients = positionalGradients;
    }

    public override void Clip(ILocalClippingStrategy<float> clipping)
    {
        this.TokenGradients.Clip(clipping);
        this.PositionalGradients.Clip(clipping);
    }

    public override IEnumerable<Tensor<float>> EnumerateParameterGradients()
    {
        foreach (var grad in TokenGradients.EnumerateParameterGradients())
            yield return grad;
        foreach (var grad in PositionalGradients.EnumerateParameterGradients())
            yield return grad;
    }
}

public class TokenPositionEmbedding : INetworkModule, IBlockVisitable
{
    public DenseLinear TokenEmbedding {get; init;}
    public DenseLinear PositionalEmbedding {get; init;}

    public int VocabSize {get; init;}
    public int MaxSequenceLength {get; init;}
    public int EmbeddingDim {get; init;}

    public int SubmoduleCount => 2;

    public TokenPositionEmbedding(int vocabSize, int maxSeqLen, int embeddingDim)
    {
        this.VocabSize = vocabSize;
        this.MaxSequenceLength = maxSeqLen;
        this.EmbeddingDim = embeddingDim;

        this.TokenEmbedding = new DenseLinear(VocabSize, EmbeddingDim);
        this.PositionalEmbedding = new DenseLinear(MaxSequenceLength, EmbeddingDim);
    }

    public void Initialize(IInitializer initializer)
    {
        TokenEmbedding.Initialize(initializer);
        PositionalEmbedding.Initialize(initializer);
    }

    public Shape ForwardShape(Shape input)
    {
        return new Shape(input[0], input[1], EmbeddingDim);
    }

    public Tensor<float> Forward(Tensor<float> input, EvaluationContext? ctx = null, ISteppedProgress? progress = null)
    {
        // Ensure [batch, windowLen] shape where each token is just an token index
        input = input.ReshapeShared(input.Shape.NormalizeRank(2));
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];

        // Input is assumed to be indices and converted to one-hot
        Tensor<float> onehot = OneHot(input, this.VocabSize);

        // Create positional indices
        Tensor<float> posOneHot = CreatePositionOneHot(batch, seqLen, this.MaxSequenceLength);

        // Evaluate
        var tokenEmbedding = this.TokenEmbedding.Forward(onehot, ctx, progress);
        var posEmbedding = this.PositionalEmbedding.Forward(posOneHot, ctx, progress);

        var output = tokenEmbedding + posEmbedding;

        return output;
    }

    private static Tensor<float> OneHot(Tensor<float> indices, int depth)
    {
        int batch = indices.Shape[0];
        int seqLen = indices.Shape[1];

        var result = Tensor<float>.Zeros(new Shape(batch, seqLen, depth));

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int idx = (int)indices[b, t];
                Span<float> row = result.SubtensorSpan(b, t);
                row[idx] = 1f;
            }
        }

        return result;
    }

    private static Tensor<float> CreatePositionOneHot(int batch, int seqLen, int depth)
    {
        var result = Tensor<float>.Zeros(new Shape(batch, seqLen, depth));

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                result[b, t, t] = 1f;
            }
        }

        return result;
    }

    public Gradients Backward(Tensor<float> dy, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        var tokenGrad = this.TokenEmbedding.Backward(dy, ctx, clipping);
        var posGrad = this.PositionalEmbedding.Backward(dy, ctx, clipping);

        var dToken = tokenGrad.dX;

        return new EmbeddingGradients(
            dToken,
            tokenGrad,
            posGrad
        );
    }

    public void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        if (gradients is not EmbeddingGradients grads)
            throw new ArgumentException("Expected a EmbeddingGradients object", nameof(gradients));

        this.TokenEmbedding.Update(learningRate, grads.TokenGradients, optimizer, regularization);
        this.PositionalEmbedding.Update(learningRate, grads.PositionalGradients, optimizer, regularization);
    }

    public TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg)
    {
        return visitor.Visit(this, arg);
    }
}*/

public class EmbeddingGradients : Gradients
{
    public Tensor<float> TokenGradients { get; }
    public Tensor<float> PositionalGradients { get; }

    public EmbeddingGradients(
        Tensor<float> dX,
        Tensor<float> tokenGradients,
        Tensor<float> positionalGradients
        ): base(dX)
    {
        TokenGradients = tokenGradients;
        PositionalGradients = positionalGradients;
    }

    public override void Clip(ILocalClippingStrategy<float> clipping)
    {
        clipping.ClipWeight(this.TokenGradients);
        clipping.ClipWeight(this.PositionalGradients);
    }

    public override IEnumerable<Tensor<float>> EnumerateParameterGradients()
    {
        yield return TokenGradients;
        yield return PositionalGradients;
    }
}

public class EmbeddingContext : IModuleContext
{
    public Tensor<float> Input {get; init;}

    public Tensor<float> Output {get; init;}

    public EmbeddingContext(Tensor<float> input, Tensor<float> output)
    {
        this.Input = input; 
        this.Output = output;
    }
}

public class LearnedEmbedding : INetworkModule, IBlockVisitable
{
    public Tensor<float> TokenEmbedding {get; set;}
    public Tensor<float> PositionalEmbedding {get; set;}

    public int TrainableParameterCount() => TokenEmbedding.ElementCount + PositionalEmbedding.ElementCount;
    public int UnTrainableParameterCount() => 0;

    public int VocabSize {get; init;}
    public int MaxSequenceLength {get; init;}
    public int EmbeddingDim {get; init;}

    public int SubmoduleCount => 0;

    public LearnedEmbedding(int vocabSize, int maxSeqLen, int embeddingDim)
    {
        this.VocabSize = vocabSize;
        this.MaxSequenceLength = maxSeqLen;
        this.EmbeddingDim = embeddingDim;

        this.TokenEmbedding = Tensor<float>.Zeros(new Shape(VocabSize, EmbeddingDim));
        this.PositionalEmbedding = Tensor<float>.Zeros(new Shape(MaxSequenceLength, EmbeddingDim));
    }

    public void Initialize(IInitializer initializer)
    {
        var parameters = TokenEmbedding.ElementCount + PositionalEmbedding.ElementCount;

        TokenEmbedding.FillGenerated(() => initializer.RandomWeight(VocabSize, EmbeddingDim, parameters));
        PositionalEmbedding.FillGenerated(() => initializer.RandomWeight(MaxSequenceLength, EmbeddingDim, parameters));
    }

    public Shape ForwardShape(Shape input)
    {
        return new Shape(input[0], input[1], EmbeddingDim);
    }

    public Tensor<float> Forward(Tensor<float> input, EvaluationContext? ctx = null, ISteppedProgress? progress = null)
    {
        // Ensure [batch, windowLen] shape where each token is just an token index
        input = input.ReshapeShared(input.Shape.NormalizeRank(2));
        
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];

        var output = Tensor<float>.Zeros(new Shape(batch, seqLen, EmbeddingDim));

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int tokenIdx = (int)input[b, t];

                if (tokenIdx < 0 || tokenIdx >= VocabSize)
                    throw new ArgumentOutOfRangeException(nameof(input), $"Token index {tokenIdx} out of range");

                if (t >= MaxSequenceLength)
                    throw new ArgumentOutOfRangeException(nameof(input), $"Sequence length exceeds max positional embedding");

                Span<float> tokenVec = TokenEmbedding.SubtensorSpan(tokenIdx);
                Span<float> posVec   = PositionalEmbedding.SubtensorSpan(t);
                Span<float> dst      = output.SubtensorSpan(b, t);

                for (int i = 0; i < EmbeddingDim; i++)
                    dst[i] = tokenVec[i] + posVec[i];
            }
        }

        if (ctx is not null) {
            ctx.Save(this, new EmbeddingContext(input, output));
        }

        return output;
    }


    public Gradients Backward(Tensor<float> dy, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        var context = ctx.Get<EmbeddingContext>(this);
        var input = context.Input;

        int batch = input.Shape[0];
        int seqLen = input.Shape[1];

        var dToken = Tensor<float>.Zeros(TokenEmbedding.Shape);
        var dPos   = Tensor<float>.Zeros(PositionalEmbedding.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int tokenIdx = (int)input[b, t];

                Span<float> gradOut = dy.SubtensorSpan(b, t);
                Span<float> tokenGrad = dToken.SubtensorSpan(tokenIdx);
                Span<float> posGrad   = dPos.SubtensorSpan(t);

                for (int i = 0; i < EmbeddingDim; i++)
                {
                    tokenGrad[i] += gradOut[i];
                    posGrad[i]   += gradOut[i];
                }
            }
        }

        return new EmbeddingGradients(
            dX: Tensor<float>.ZerosLike(input),              // no meaningful gradient wrt indices
            tokenGradients: dToken,
            positionalGradients: dPos
        );
    }

    public void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        if (gradients is not EmbeddingGradients grads)
            throw new ArgumentException("Expected a EmbeddingGradients object", nameof(gradients));

        var dW1 = grads.TokenGradients;
        var dW2 = grads.PositionalGradients;

        // Apply regularization to weights
        if (regularization is not null)
        {
            dW1.ElementWiseBinaryInplace(this.TokenEmbedding, (gradient, prevWeight) => gradient + regularization.Invoke(prevWeight));
            dW2.ElementWiseBinaryInplace(this.PositionalEmbedding, (gradient, prevWeight) => gradient + regularization.Invoke(prevWeight));
        }

        // Apply optimizer
        optimizer.UpdateParameter(this, nameof(TokenEmbedding), learningRate, this.TokenEmbedding, dW1);
        optimizer.UpdateParameter(this, nameof(PositionalEmbedding), learningRate, this.PositionalEmbedding, dW2);
    }

    public TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg)
    {
        return visitor.Visit(this, arg);
    }
}