namespace DotML.Network.Embedding;

/// <summary>
/// Behaviour of an object that can tokenize an input string
/// </summary>
/// <typeparam name="TToken">token type</typeparam>
public interface ITokenizer<TToken> where TToken:notnull
{
    /// <summary>
    /// Tokenize an input corpus of text
    /// </summary>
    /// <param name="corpus">input string</param>
    /// <returns>enumerable of tokens</returns>
    public IEnumerable<TToken> Tokenize(string corpus);
}