namespace DotML.Network.IO.Netbuild;

public class BuildEnvironment {
    public Shape3D InputShape;
    public Dictionary<string, object> Arguments {get; private set;} = new Dictionary<string, object>();
    public Dictionary<string, int> LayerAliases {get; private set;} = new Dictionary<string, int>();
    public FeedforwardNetwork? Network {get; set;}
    public NetbuildSerializer? Serializer {get; set;}
    public Dictionary<string, Func<string>>? ScopedNetworks {get; set;}
}