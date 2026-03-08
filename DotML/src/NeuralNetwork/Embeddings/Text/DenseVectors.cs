namespace DotML.Network.Embedding.Text;

public interface IPositionEncoder
{
    public int PositionEncodingLength {get;}
    public void EmbedPosition(int position, Span<float> vector);
}

public class AddSinusoidalPositionEncoder : IPositionEncoder
{    
    public int PositionEncodingLength => 0; // Length 0, because we are doing elementwise addition we need no additional space in the vector reserved for the position
    public void EmbedPosition(int position, Span<float> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            double angle = position / Math.Pow(10000, 2.0 * (i / (double)vector.Length));
            vector[i] += i % 2 == 0 ? (float)Math.Sin(angle) : (float)Math.Cos(angle);
        }
    }
}

public class ConcatSinusoidalPositionEncoder : IPositionEncoder
{    
    public int PositionEncodingLength {get; init;} // Since we are concating the position, we need to reserve additional space for the position information

    public ConcatSinusoidalPositionEncoder(int length)
    {
        this.PositionEncodingLength = Math.Max(0, length);
    }

    public void EmbedPosition(int position, Span<float> vector)
    {
        // Determine where to start modifying in the vector preserving everything before the offset (ie token dense encoding)
        var offset = Math.Max(0, vector.Length - PositionEncodingLength);

        // Modify the region reserved for position encoding
        for (int i = offset; i < vector.Length; i++)
        {
            double angle = position / Math.Pow(10000, 2.0 * (i / (double)vector.Length));
            vector[i] += i % 2 == 0 ? (float)Math.Sin(angle) : (float)Math.Cos(angle);
        }
    }
}

/// <summary>
/// Dense-vector embedding where each token maps to a separate dense vector. It can additionally add positional encoding to the output vectors.
/// </summary>
/// <typeparam name="TToken">token type</typeparam>
public class DenseVectors<TToken> : IEmbedding<IEnumerable<TToken>, float> where TToken : notnull
{
    protected readonly IReadOnlyDictionary<TToken, Vec<float>> Vocab;

    public int VocabLength => Vocab.Count;
    public int EmbeddingVectorLength {get; init;}

    public IPositionEncoder? PositionEncoder {get; set;}

    public DenseVectors(IPositionEncoder? positional, params IEnumerable<(TToken Token, Vec<float> DenseVector)> vocab): this(vocab)
    {
        // The vocab (supported characters)
        this.PositionEncoder = positional;
    }

    public DenseVectors(params IEnumerable<(TToken Token, Vec<float> DenseVector)> vocab)
    {
        // The vocab (supported characters)
        var cab = new Dictionary<TToken, Vec<float>>();
        int len = 0;
        foreach (var v in vocab) {
            cab[v.Token] = v.DenseVector;
            len = Math.Max(len, v.DenseVector.Dimensionality);
        }
        this.Vocab = cab;
        this.EmbeddingVectorLength = len;
    }

    /// <summary>
    /// Convert a value to a tensor embedding of shape [len(sequence), len(embedding_vector)]
    /// </summary>
    /// <param name="value">value to convert</param>
    /// <returns>tensor representation of the value</returns>
    public Tensor<float> ToTensor(IEnumerable<TToken> value)
    {
        var toks = value.ToList();

        var concatLength = this.PositionEncoder?.PositionEncodingLength ?? 0;
        var shape = new Shape(toks.Count, EmbeddingVectorLength + concatLength);
        var embedding = Tensor<float>.Zeros(shape);
        int row = 0;
        foreach (var tok in toks)
        {
            if (!Vocab.TryGetValue(tok, out var vec))
            {
                row++;
                continue;
            }

            var embedding_row = embedding.SubtensorSpan(row);
            var vec_row = vec.AsSpan();
            vec_row.CopyTo(embedding_row);

            PositionEncoder?.EmbedPosition(row, embedding_row);

            row++;
            continue;
        }

        return embedding;
    }
}

public interface IDenseVectorGenerator<TToken>
{
    public Vec<float> Generate(int tokenId, int vectorLength, TToken token);
}

public class OneHotVectorGenerator<TToken> : IDenseVectorGenerator<TToken>
{
    public Vec<float> Generate(int tokenId, int vectorLength, TToken token)
    {
        var vec = new Vec<float>(vectorLength);
        if (tokenId >= 0 && tokenId < vectorLength)
            vec[tokenId] = 1;
        return vec;
    }
}

/// <summary>
/// One hot encoding of tokens to vectors where each vector is sparse with a 1 at a specific unique index for each type of token forming a series of unique basis vectors
/// </summary>
/// <typeparam name="TToken">token type</typeparam>
public class OneHot<TToken> : DenseVectors<TToken> where TToken:notnull
{
    private static OneHotVectorGenerator<TToken> generator = new OneHotVectorGenerator<TToken>();
    private static IEnumerable<(TToken, Vec<float>)> generateOneHot(IEnumerable<TToken> tokens)
    {
        var pairs = tokens.Distinct().Index().ToList(); // Doing this because I need the count
        foreach (var p in pairs)
        {
            var tok = p.Item;
            var index = p.Index;

            yield return (tok, generator.Generate(index, pairs.Count, tok));
        }
    }

    public OneHot(params IEnumerable<TToken> tokens): base(generateOneHot(tokens)) {}
}

public class TrigHashDenseVectorGenerator<TToken>: IDenseVectorGenerator<TToken> where TToken:notnull
{
    public Vec<float> Generate(int tokenId, int vectorLength, TToken token)
    {
        var vec = new Vec<float>(vectorLength);
        int hash = token.GetHashCode();

        for (int i = 0; i < vectorLength; i++)
        {
            double angle = hash / Math.Pow(10000, 2.0 * (i / (double)vectorLength));
            vec[i] = (float)(Math.Sin(angle) + Math.Cos(angle * 1.37));
        }

        return vec;
    }
}

public class FourierVectorGenerator<TToken> : IDenseVectorGenerator<TToken>
{
    public float Frequency {get; init;}
    public FourierVectorGenerator(float freq)
    {
        Frequency = Math.Max(1, freq);
    }

    public Vec<float> Generate(int tokenId, int vectorLength, TToken token)
    {
        var vec = new Vec<float>(vectorLength);
        for (var i = 0; i < vectorLength; i++)
        {
            var (quotient, remainder) = Math.DivRem(i, 2);
            vec[i] = remainder == 0 
                ? MathF.Sin(2 * (quotient + 2) * Frequency * tokenId * MathF.PI/vectorLength) 
                : MathF.Cos(2 * (quotient + 2) * Frequency * tokenId * MathF.PI/vectorLength);
        }
        return vec;
    }
}