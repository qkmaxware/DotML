namespace DotML;

/// <summary>
/// Any object that is serializable to JSON
/// </summary>
public interface IJsonizable {
    /// <summary>
    /// Convert this object to a JSON representation
    /// </summary>
    public void ToJson(TextWriter writer);

    /// <summary>
    /// Convert this object to a JSON representation
    /// </summary>
    /// <returns>JSON serialized string</returns>
    public string ToJson() {
        using var writer = new StringWriter();
        ToJson(writer);
        return writer.ToString();
    }

    /// <summary>
    /// Convert this object to a JSON representation, or a default value if serialization fails
    /// </summary>
    /// <param name="default">default value</param>
    /// <returns>JSON string or default string</returns>
    public string ToJsonOrDefault(string @default) {
        try {
            return ToJson();
        } catch {
            return @default;
        }
    }
}