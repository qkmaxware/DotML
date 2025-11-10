using System.Data;
using System.Reflection;

namespace DotML.Network.IO.Netbuild;

public class ArgumentMap : System.Collections.Generic.Dictionary<string, DotML.Network.IO.Netbuild.Literal> {
    public BuildEnvironment Env {get; private set;}
    public ArgumentMap(BuildEnvironment environment) {
        this.Env = environment;
    }
    public ArgumentMap(BuildEnvironment environment, System.Collections.Generic.Dictionary<string, DotML.Network.IO.Netbuild.Literal> args) : base(args) {
        this.Env = environment;
    }
}