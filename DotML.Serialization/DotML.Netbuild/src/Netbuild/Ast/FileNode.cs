namespace DotML.Network.IO.Netbuild;

public delegate void NetbuildStatementHandler(int current_statement_index, int total_statement_count, Statement? current_statement);

public class File : AstNode
{
    public Preamble Preamble { get; private set; }
    public List<Statement> Statements { get; private set; } = new List<Statement>();

    public File(Preamble preamble)
    {
        this.Preamble = preamble;
    }
    
    public INetworkModule MakeModule(BuildEnvironment env, NetbuildStatementHandler? stmt_action = null)
    {
        var statment_count = 1 + Statements.Count;
        stmt_action?.Invoke(0, statment_count, Preamble.From);
        Preamble.From?.ModuleAction(env);
        if (env.NetworkBlock is null)
            return new SequentialBlock();

        var stmt_index = 1;
        foreach (var command in Statements)
        {
            stmt_action?.Invoke(stmt_index++, statment_count, command);
            command.ModuleAction(env);
        }

        return env.NetworkBlock;
    }
}