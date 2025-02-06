namespace DotML;

/// <summary>
/// Any object that is able to be drawn as a diagram
/// </summary>
public interface IDiagrammable {
    /// <summary>
    /// Convert this object to an SVG formatted diagram
    /// </summary>
    public void ToSvg(TextWriter writer);

    /// <summary>
    /// Convert this object to an SVG formatted diagram
    /// </summary>
    /// <returns>SVG string</returns>
    public string ToSvg() {
        using var writer = new StringWriter();
        ToSvg(writer);
        return writer.ToString();
    }
}