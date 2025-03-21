namespace DotML.Network.IO.Netbuild;

public class ReplaceStatement : Statement {
    LayerReference reference;
    private string layer_name;
    Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory;
    public Dictionary<string, Literal> arguments = new Dictionary<string, Literal>();

    public ReplaceStatement(LayerReference reference, string layer_name, Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory, List<(Token<string>, Literal)> args) {
        this.factory = factory;
        foreach (var pair in args) {
            arguments[pair.Item1.Value] = pair.Item2;
        } 
        this.layer_name = layer_name;
        this.reference = reference;
    }

    public override void Action(BuildEnvironment env) {
        var network = env.Network;
        if (network is null)
            return;
        
        var replacement_index = reference.IndexOf(env.LayerAliases);
        var input_shape = network.GetLayer(replacement_index).InputShape;
        IFeedforwardNetworkLayer layer = factory(input_shape, arguments);
        network.ReplaceLayer(replacement_index, layer);
    }

    public override string ToString() {
        return $"REPLACE {reference} WITH {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))}";
    }
}