using CommandLine;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;
using System.Reflection;

namespace DotML.Examples;

[Verb("test", HelpText = "Test a trained network against its validation dataset")]
public class Validate : Command
{
    [Option("raws", HelpText = "Set flag to indicate that validation data is in a raw format and needs preprocessing before validation")]
    public bool ProcessRaws { get; set; }

    public override void Exec(IExample example)
    {
        if (ProcessRaws)
            example.ProcessRawData();
        
        example.Validate();
    }
}