namespace DotML.Network.IO.Netbuild;

public class InsertBeforeStatement : Statement {
    private LayerReference reference;
    string layer_name;
    public Dictionary<string, Literal> arguments = new Dictionary<string, Literal>();

    public InsertBeforeStatement(LayerReference reference, string layer_name, List<(Token<string>, Literal)> args) {
        this.reference = reference;
        this.layer_name = layer_name;
        foreach (var pair in args) {
            arguments[pair.Item1.Value] = pair.Item2;
        } 
    }

    public override void ModuleAction(BuildEnvironment env)
    {
        var network = env.NetworkBlock;
        if (network is null)
            return;
        
        // Find the layer to insert after
        var layer_index = this.reference.IndexOf(env.LayerAliases);
        var layer = network.GetLayer(layer_index);
        if (layer is null)
            throw new ArgumentException($"Unknown layer '{layer_name}'");

        // Create the layer 
        var output_shape = network.LayerCount > 0 ? network.ForwardShapeUntil(env.InputShape, layer_index - 1) : env.InputShape;
        INetworkModule? module = AddStatement.makeLayer(this.layer_name, output_shape, new ArgumentMap(env, arguments));
        var index = layer_index + 1;

        if (module is not null)
            network.InsertBefore(layer, module);
    }

    public override string ToString()
    {
        return $"INSERT BEFORE {reference} {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))}";
    }
}

internal class InsertAfterStatement : Statement {
    private LayerReference reference;
    string layer_name;
    public Dictionary<string, Literal> arguments = new Dictionary<string, Literal>();

    public InsertAfterStatement(LayerReference reference, string layer_name, List<(Token<string>, Literal)> args) {
        this.reference = reference;
        this.layer_name = layer_name;
        foreach (var pair in args) {
            arguments[pair.Item1.Value] = pair.Item2;
        } 
    }

    public override void ModuleAction(BuildEnvironment env)
    {
        var network = env.NetworkBlock;
        if (network is null)
            return;
        
        // Find the layer to insert after
        var layer_index = this.reference.IndexOf(env.LayerAliases);
        var layer = network.GetLayer(layer_index);
        if (layer is null)
            throw new ArgumentException($"Unknown layer '{layer_name}'");

        // Create the layer 
        var output_shape = network.LayerCount > 0 ? network.ForwardShapeUntil(env.InputShape, layer_index) : env.InputShape;
        INetworkModule? module = AddStatement.makeLayer(this.layer_name, output_shape, new ArgumentMap(env, arguments));
        var index = layer_index + 1;

        if (module is not null)
            network.InsertAfter(layer, module);
    }

    public override string ToString()
    {
        return $"INSERT AFTER {reference} {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))}";
    }
}