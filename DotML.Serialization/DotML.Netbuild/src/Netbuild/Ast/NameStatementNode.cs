using DotML.Network.IO.Netbuild;

namespace DotML.Network.IO.Netbuild;

public class NameStatement : Statement {
    private string name;
    public NameStatement(string name) {
        this.name = name;
    }
    
    public override void ModuleAction(BuildEnvironment env) {
        var net = env.NetworkBlock;
        if (net is not null)
            net.Alias = name;
    }

    public override string ToString()
    {
        return $"NAME {name}";
    }
}