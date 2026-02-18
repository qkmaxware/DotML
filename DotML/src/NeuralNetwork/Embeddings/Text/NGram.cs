
using System.Text.RegularExpressions;

namespace DotML.Network.Embedding.Text;

/// <summary>
/// N-Gram style embedding where character ordering is considered in windows of size N across each corpus. 
/// </summary>
public class CharacterNGram : IEmbedding<string, int>
{
    private readonly Dictionary<char, int> Vocab;
    public int N {get; init;}

    public int VocabLength => Vocab.Count;
    public readonly TensorShape TensorShape;

    /// <summary>
    /// Create a new N-Gram embedding with the given window size and vocab
    /// </summary>
    /// <param name="n">adjacency window size</param>
    /// <param name="vocab">words in the vocabulary</param>
    public CharacterNGram(int n, params IEnumerable<string> vocab)
    {
        // The N
        this.N = Math.Max(1, n);

        // The vocab (supported characters)
        int charIndex = 0;
        this.Vocab = new Dictionary<char, int>();
        foreach (var word in vocab)
        {
            foreach (var letter in word)
            {
                if (!Vocab.ContainsKey(letter))
                {
                    Vocab.Add(letter, charIndex++);
                }
            }
        }

        // The shape of an output tensor
        var shape = new int[N];
        for (int i = 0; i < N; i++)
            shape[i] = VocabLength;
        this.TensorShape = new TensorShape(shape); // This is N-D, should it be a 1D flattened tensor?
    }

    /// <summary>
    /// Create an N-Gram embedding with an N of 2.
    /// </summary>
    /// <param name="vocab">words in the vocabulary</param>
    /// <returns>N-Gram embedding</returns>
    public static CharacterNGram Bigram(params IEnumerable<string> vocab) => new CharacterNGram(2, vocab);

    /// <summary>
    /// Create an N-Gram embedding with an N of 3.
    /// </summary>
    /// <param name="vocab">words in the vocabulary</param>
    /// <returns>N-Gram embedding</returns>
    public static CharacterNGram Trigram(params IEnumerable<string> vocab) => new CharacterNGram(3, vocab);

    public int IndexOf(char c) {
        if (Vocab.TryGetValue(c, out int index))
        {
            return index;
        } else
        {
            return -1;
        }
    }

    public Tensor<int> ToTensor(string value)
    {
        var embedding = Tensor<int>.Zeros(this.TensorShape);

        if (string.IsNullOrEmpty(value))
            return embedding;

        Span<int> coordinates = stackalloc int[N];      // Coordinates for indexing into the tensor
        for (int i = 0; i < value.Length; i++)
        {
            int filled = 0;
            int strIndex = i;

            // Build index tuple
            while (filled < N && strIndex < value.Length)
            {
                char c = value[strIndex++];
                if (!Vocab.TryGetValue(c, out int index))
                    continue;

                coordinates[filled++] = index;
            }

            // If built successfully, increment the count
            if (filled == N)
                embedding[coordinates]  += 1;
        }

        return embedding;
    }
}

/// <summary>
/// N-Gram style embedding where word ordering is considered in windows of size N across each corpus. 
/// </summary>
public class WordNGram : IEmbedding<string, int>
{
    private readonly Dictionary<string, int> Vocab;
    public int N {get; init;}

    public int VocabLength => Vocab.Count;
    public readonly TensorShape TensorShape;

    /// <summary>
    /// Create a new N-Gram embedding with the given window size and vocab
    /// </summary>
    /// <param name="n">adjacency window size</param>
    /// <param name="vocab">words in the vocabulary</param>
    public WordNGram(int n, params IEnumerable<string> vocab)
    {
        // The N
        this.N = Math.Max(1, n);

        // The vocab (supported characters)
        this.Vocab = vocab.Distinct().Index().ToDictionary(x => x.Item, x => x.Index);

        // The shape of an output tensor
        var shape = new int[N];
        for (int i = 0; i < N; i++)
            shape[i] = VocabLength;
        this.TensorShape = new TensorShape(shape); // This is N-D, should it be a 1D flattened tensor?
    }

    /// <summary>
    /// Create an N-Gram embedding with an N of 2.
    /// </summary>
    /// <param name="vocab">words in the vocabulary</param>
    /// <returns>N-Gram embedding</returns>
    public static WordNGram Bigram(params IEnumerable<string> vocab) => new WordNGram(2, vocab);

    /// <summary>
    /// Create an N-Gram embedding with an N of 3.
    /// </summary>
    /// <param name="vocab">words in the vocabulary</param>
    /// <returns>N-Gram embedding</returns>
    public static WordNGram Trigram(params IEnumerable<string> vocab) => new WordNGram(3, vocab);

    public int IndexOf(string word) {
        if (Vocab.TryGetValue(word, out int index))
        {
            return index;
        } else
        {
            return -1;
        }
    }

    private static Regex wordPattern = new Regex(@"\b[a-zA-Z0-9]+(?:[-’'][a-zA-Z0-9]+)*\b");

    public Tensor<int> ToTensor(string value)
    {
        var embedding = Tensor<int>.Zeros(this.TensorShape);

        if (string.IsNullOrEmpty(value))
            return embedding;

        var matches = wordPattern.Matches(value);
        if (matches.Count == 0)
            return embedding;

        Span<int> coordinates = stackalloc int[N];      // Coordinates for indexing into the tensor
        for (int i = 0; i < matches.Count; i++)
        {
            int filled = 0;
            int matchIndex = i;

            // Build index tuple
            while (filled < N && matchIndex < matches.Count)
            {
                var match = matches[matchIndex++];
                if (!match.Success || !Vocab.TryGetValue(match.Value, out int index))
                    continue;

                coordinates[filled++] = index;
            }

            // If built successfully, increment the count
            if (filled == N)
                embedding[coordinates]  += 1;
        }

        return embedding;
    }
}

/// <summary>
/// N-Gram style embedding where token ordering is considered in windows of size N across each corpus. 
/// </summary>
public class TokenNGram<TToken> : IEmbedding<IEnumerable<TToken>, int> where TToken:notnull
{
    private readonly Dictionary<TToken, int> Vocab;
    public int N {get; init;}

    public int VocabLength => Vocab.Count;
    public readonly TensorShape TensorShape;

    /// <summary>
    /// Create a new N-Gram embedding with the given window size and vocab
    /// </summary>
    /// <param name="n">adjacency window size</param>
    /// <param name="vocab">tokens in the vocabulary</param>
    public TokenNGram(int n, params IEnumerable<TToken> vocab)
    {
        // The N
        this.N = Math.Max(1, n);

        // The vocab (supported characters)
        this.Vocab = vocab.Distinct().Index().ToDictionary(x => x.Item, x => x.Index);

        // The shape of an output tensor
        var shape = new int[N];
        for (int i = 0; i < N; i++)
            shape[i] = VocabLength;
        this.TensorShape = new TensorShape(shape); // This is N-D, should it be a 1D flattened tensor?
    }

    /// <summary>
    /// Create an N-Gram embedding with an N of 2.
    /// </summary>
    /// <param name="vocab">words in the vocabulary</param>
    /// <returns>N-Gram embedding</returns>
    public static TokenNGram<TToken> Bigram(params IEnumerable<TToken> vocab) => new TokenNGram<TToken>(2, vocab);

    /// <summary>
    /// Create an N-Gram embedding with an N of 3.
    /// </summary>
    /// <param name="vocab">words in the vocabulary</param>
    /// <returns>N-Gram embedding</returns>
    public static TokenNGram<TToken> Trigram(params IEnumerable<TToken> vocab) => new TokenNGram<TToken>(3, vocab);

    public int IndexOf(TToken token) {
        if (Vocab.TryGetValue(token, out int index))
        {
            return index;
        } else
        {
            return -1;
        }
    }

    /// <summary>
    /// Convert a value to a tensor embedding of shape [len(Vocab), ..., len(Vocab)] with N dimensions
    /// </summary>
    /// <param name="value">value to convert</param>
    /// <returns>tensor representation of the value</returns>
    public Tensor<int> ToTensor(IEnumerable<TToken> value)
    {
        var embedding = Tensor<int>.Zeros(this.TensorShape);

        Span<int> coordinates = stackalloc int[N];      // Coordinates for indexing into the tensor
        int filled = 0;

        foreach (var token in value)
        {
            if (!Vocab.TryGetValue(token, out int index))
                continue;  

            // Shift left
            for (int i = 0; i < N - 1; i++)
            {
                coordinates[i] = coordinates[i + 1];
            }

            coordinates[N - 1] = index;

            if (filled < N)
            {
                filled++;
                continue;
            }

            embedding[coordinates] += 1;
        }

        return embedding;
    }

    /// <summary>
    /// Convert a value to a tensor embedding of shape [len(Vocab), ..., len(Vocab)] with N dimensions
    /// </summary>
    /// <param name="value">value to convert</param>
    /// <returns>tensor representation of the value</returns>
    public Tensor<int> CreateEmbedding(ReadOnlySpan<TToken> value)
    {
        var embedding = Tensor<int>.Zeros(this.TensorShape);
        
        Span<int> coordinates = stackalloc int[N];      // Coordinates for indexing into the tensor
        int filled = 0;

        foreach (var token in value)
        {
            if (!Vocab.TryGetValue(token, out int index))
                continue;  

            // Shift left
            for (int i = 0; i < N - 1; i++)
            {
                coordinates[i] = coordinates[i + 1];
            }

            coordinates[N - 1] = index;

            if (filled < N)
            {
                filled++;
                continue;
            }

            embedding[coordinates] += 1;
        }

        return embedding;
    }
}