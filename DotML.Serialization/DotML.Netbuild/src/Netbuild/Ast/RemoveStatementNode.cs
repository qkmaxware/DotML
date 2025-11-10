namespace DotML.Network.IO.Netbuild;

public class RemoveStatement : Statement {
    private LayerReference reference;
    public RemoveStatement(LayerReference reference) {
        this.reference = reference;
    }
    
    public override void ModuleAction(BuildEnvironment env) {
        var network = env.NetworkBlock;
        if (network is null)
            return;
        
        network.RemoveAt(reference.IndexOf(env.LayerAliases));
    }

    public override string ToString()
    {
        return $"REMOVE {reference}";
    }
}