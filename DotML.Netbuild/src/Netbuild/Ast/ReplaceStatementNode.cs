namespace DotML.Network.IO.Netbuild;

public class ReplaceStatement : Statement {
    LayerReference reference;
    private string layer_name;
    public Dictionary<string, Literal> arguments = new Dictionary<string, Literal>();

    public ReplaceStatement(LayerReference reference, string layer_name, List<(Token<string>, Literal)> args) {
        foreach (var pair in args) {
            arguments[pair.Item1.Value] = pair.Item2;
        } 
        this.layer_name = layer_name;
        this.reference = reference;
    }
    
    public override void ModuleAction(BuildEnvironment env) {
        var network = env.NetworkBlock;
        if (network is null)
            return;
        
        var replacement_index = reference.IndexOf(env.LayerAliases);
        var original = network.GetLayer(replacement_index);
        var output_shape = network.LayerCount > 0 ? network.ForwardShapeUntil(env.InputShape, replacement_index - 1) : env.InputShape;
        INetworkModule? module = AddStatement.makeLayer(this.layer_name, output_shape, new ArgumentMap(env, arguments));

        if (module is not null)
            network.Replace(original, module);
    }

    public override string ToString()
    {
        return $"REPLACE {reference} WITH {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))}";
    }
}