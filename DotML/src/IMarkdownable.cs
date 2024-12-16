namespace DotML.Network;

/// <summary>
/// Any object that is serializable to Markdown
/// </summary>
public interface IMarkdownable {
    /// <summary>
    /// Convert this object to a Markdown representation
    /// </summary>
    /// <returns>Markdown serialized string</returns>
    public string ToMarkdown();
}

/// <summary>
/// Any object that is serializable to HTML
/// </summary>
public interface IHtmlable {
    /// <summary>
    /// Convert this object to an HTML representation
    /// </summary>
    /// <returns>HTML serialized string</returns>
    public string ToHtml();
}