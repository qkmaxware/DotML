namespace DotML.Network.IO.Netbuild;

public class BuildEnvironment {
    public Shape3D InputShape;
    public Dictionary<string, object> Arguments {get; private set;} = new Dictionary<string, object>();
    public Dictionary<string, int> LayerAliases {get; private set;} = new Dictionary<string, int>();
    public FeedforwardNetwork? Network {get; set;}
    public NetbuildSerializer? Serializer {get; set;}
    public Dictionary<string, Func<string>>? ScopedNetworks {get; set;}

    public IFeedforwardNetworkLayer? GetLayer(string alias) {
        if (!LayerAliases.TryGetValue(alias, out int index)) {
            return null;
        }
        return Network is not null && index >= 0 && index < Network.LayerCount ? Network.GetLayer(index) : null;
    }
}