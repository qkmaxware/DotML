using System.Collections;

public static class EnumeratorExtensions
{
    /// <summary>
    /// Advance the enumerator to the end of the sequence
    /// </summary>
    public static void MoveEnd<T>(this IEnumerator<T> self)
    {
        while (self.MoveNext()) { }
    }
}