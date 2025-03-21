using DotML.Network.IO.Netbuild;

namespace DotML.Network.IO.Netbuild;

public class NameStatement : Statement {
    private string name;
    public NameStatement(string name) {
        this.name = name;
    }

    public override void Action(BuildEnvironment env) {
        var net = env.Network;
        if (net is not null)
            net.Name = name;
    }

    public override string ToString() {
        return $"NAME {name}";
    }
}