namespace DotML.Network.IO.Netbuild;

public abstract class Statement : AstNode
{
    [Obsolete("Use ModuleAction instead")]
    public abstract void Action(BuildEnvironment env);
    public abstract void ModuleAction(BuildEnvironment env);
}