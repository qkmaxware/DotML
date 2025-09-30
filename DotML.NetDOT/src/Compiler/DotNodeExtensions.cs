using DotML.NetDot.Dot;

namespace DotML.NetDot;

internal static class DotNodeExtensions
{
    public static bool IsInput(this DotVertex node)
    {
        return node.Attributes.TryGetValue("type", out var type_name) && string.Equals(type_name, "input", StringComparison.InvariantCultureIgnoreCase);
    }

    public static bool IsOutput(this DotVertex node)
    {
        return node.Attributes.TryGetValue("type", out var type_name) && string.Equals(type_name, "output", StringComparison.InvariantCultureIgnoreCase);
    }
}
