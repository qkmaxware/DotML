using CommandLine;

namespace DotML.Examples;

public class Program
{
    public static int Main()
    {
        return Parser
            .Default
            .ParseArguments<List, Run, Train, Clean>(Environment.GetCommandLineArgs().Skip(1))
            .MapResult(
                (List lst) => lst.TryExec(),
                (Run run) => run.TryExec(),
                (Train train) => train.TryExec(),
                (Clean clean) => clean.TryExec(),
                _ => 1
            );
    }
}