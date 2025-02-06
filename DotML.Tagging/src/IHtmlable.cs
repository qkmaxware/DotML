namespace DotML.Network;

/// <summary>
/// Any object that is serializable to HTML
/// </summary>
public interface IHtmlable {
    /// <summary>
    /// Convert this object to an HTML representation
    /// </summary>
    public void ToHtml(TextWriter writer);

    /// <summary>
    /// Convert this object to an SVG formatted diagram
    /// </summary>
    /// <returns>SVG string</returns>
    public string ToHtml() {
        using var writer = new StringWriter();
        ToHtml(writer);
        return writer.ToString();
    }
}