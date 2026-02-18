using CommandLine;

namespace DotML.Examples;

public class Program
{
    public static int Main()
    {
        return Parser
            .Default
            .ParseArguments<About, List, Run, Train, Validate, Clean>(Environment.GetCommandLineArgs().Skip(1))
            .MapResult(
                (About about) => about.TryExec(),
                (List lst) => lst.TryExec(),
                (Run run) => run.TryExec(),
                (Train train) => train.TryExec(),
                (Validate valid) => valid.TryExec(),
                (Clean clean) => clean.TryExec(),
                _ => 1
            );
    }
}