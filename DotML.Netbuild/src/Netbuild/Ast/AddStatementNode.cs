namespace DotML.Network.IO.Netbuild;

public class AddStatement : Statement {
    string layer_name;
    Func<Shape3D, ArgumentMap, IFeedforwardNetworkLayer> factory;
    string? ident;
    public Dictionary<string, Literal> arguments = new Dictionary<string, Literal>();

    public AddStatement(string layer_name, Func<Shape3D, ArgumentMap, IFeedforwardNetworkLayer> factory, List<(Token<string>, Literal)> args, string? alias) {
        this.layer_name = layer_name;
        this.factory = factory;
        foreach (var pair in args) {
            arguments[pair.Item1.Value] = pair.Item2;
        } 
        this.ident = alias;
    }


    public override void Action(BuildEnvironment env) {
        var network = env.Network;
        if (network is null)
            return;
        
        var output_shape = network.LayerCount > 0 ? network.OutputShape : env.InputShape;
        IFeedforwardNetworkLayer layer = factory(output_shape, new ArgumentMap(env, arguments));
        var index = network.LayerCount;
        network.AddLayer(layer);

        if (!string.IsNullOrEmpty(ident)) {
            env.LayerAliases[ident] = index;
        }
    }

    public override string ToString() {
        if (ident is not null) {
            return $"ADD {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))} AS {ident}";
        } else {
            return $"ADD {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))}";
        }
    }
}