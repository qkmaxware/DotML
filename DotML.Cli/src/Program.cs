using CommandLine;
using DotML.Cli.Commands;

namespace DotML.Cli;

public class Program {
    public static int Main() {
        var appData = new AppData();
        
        return 
            Parser
            .Default
            .ParseArguments<Build, List, Describe, Remove, Tag, Fit, Report, Test, Run>(Environment.GetCommandLineArgs().Skip(1))
            .MapResult(
                (Build cmd)     => cmd.TryDoAction(appData),
                (List cmd)      => cmd.TryDoAction(appData),
                (Describe cmd)  => cmd.TryDoAction(appData),
                (Remove cmd)    => cmd.TryDoAction(appData),
                (Tag cmd)       => cmd.TryDoAction(appData),
                (Fit cmd)       => cmd.TryDoAction(appData),
                (Report cmd)    => cmd.TryDoAction(appData),    
                (Test cmd)      => cmd.TryDoAction(appData),
                (Run cmd)       => cmd.TryDoAction(appData),
                errs            => BaseCommand.GENERIC_ERROR
            );
    }
}