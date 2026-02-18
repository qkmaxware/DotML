namespace DotML.Network.Embedding.Text;

/// <summary>
/// Bag of words style embedding in which each occurence of a word in the vocab contributes a '1' to its representative spot in the tensor
/// </summary>
public class BagOfWords : IEmbedding<string, int>
{
    private readonly HashSet<string> Vocab;

    public int VocabLength => Vocab.Count;
    public readonly TensorShape TensorShape;

    /// <summary>
    /// Create a new bag of words embedding on a specific vocabulary of words
    /// </summary>
    /// <param name="vocab">words in the vocabulary</param>
    public BagOfWords(params IEnumerable<string> vocab)
    {
        this.Vocab = [.. vocab];
        this.TensorShape = new TensorShape(VocabLength);
    }

    private static int countOccurances(string str, string word)
    {
        int pos = 0;
        int count = 0;

        while ((pos < str.Length) && (str.IndexOf(word, pos) != -1))
        {
            count++;
            pos += word.Length;
        }

        return count;
    }

    /// <summary>
    /// Convert a value to a tensor embedding of shape [len(Vocab)]
    /// </summary>
    /// <param name="value">value to convert</param>
    /// <returns>tensor representation of the value</returns>
    public Tensor<int> ToTensor(string value)
    {
        Tensor<int> embedding = Tensor<int>.Zeros(this.TensorShape);

        foreach (var (index, word) in Vocab.Index())
        {
            embedding[index] += countOccurances(value, word);
        }

        return embedding;
    }
}