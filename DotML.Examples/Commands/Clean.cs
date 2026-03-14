using CommandLine;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Examples;

[Verb("clean", HelpText = "Cleanup temporary files associated with the example")]
public class Clean : Command
{
    
    public override void Exec(IExample example)
    {
        example.Clean();
    }

}