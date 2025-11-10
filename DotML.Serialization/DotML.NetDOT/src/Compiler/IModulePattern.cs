using DotML.NetDot.Dot;

namespace DotML.NetDot;

public interface IModulePattern
{
    public bool TryReplacePattern(DotGraph graph);
}