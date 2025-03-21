namespace DotML.Network.IO.Netbuild;

public abstract class Statement : AstNode {
    public abstract void Action(BuildEnvironment env);
}