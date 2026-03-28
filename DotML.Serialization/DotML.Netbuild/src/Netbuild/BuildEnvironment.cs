namespace DotML.Network.IO.Netbuild;

public class BuildEnvironment
{
    public Shape3D InputShape;
    public Dictionary<string, object> Arguments { get; private set; } = new Dictionary<string, object>();
    public Dictionary<string, int> LayerAliases { get; private set; } = new Dictionary<string, int>();
    public SequentialBlock? NetworkBlock { get; set; }
    public NetbuildSerializer? Serializer { get; set; }
    public Dictionary<string, Func<string>>? ScopedNetworks { get; set; }
    
    public INetworkModule? GetSubmodule(string alias)
    {
        if (!LayerAliases.TryGetValue(alias, out int index))
        {
            return null;
        }
        return NetworkBlock is not null && index >= 0 && index < NetworkBlock.LayerCount ? NetworkBlock.GetLayer(index) : null;
    }
}