namespace DotML.Network;

/// <summary>
/// Any object that is serializable to Markdown
/// </summary>
public interface IMarkdownable {
    /// <summary>
    /// Convert this object to a Markdown representation
    /// </summary>
    public void ToMarkdown(TextWriter writer);

    /// <summary>
    /// Convert this object to a Markdown representation
    /// </summary>
    /// <returns>Markdown serialized string</returns>
    public string ToMarkdown() {
        using var writer = new StringWriter();
        ToMarkdown(writer);
        return writer.ToString();
    }
}