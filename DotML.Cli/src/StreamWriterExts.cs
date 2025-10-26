using DotML.Network;

public static class StreamWriterExtensions
{
    public static void WriteSeparated(this StreamWriter writer, string separator, params ReadOnlySpan<object?> items)
    {
        for (var i = 0; i < items.Length; i++)
        {
            if (i != 0)
                writer.Write(separator);
            writer.Write(items[i]);
        }
    }
}