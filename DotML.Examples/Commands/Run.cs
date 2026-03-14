using CommandLine;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Examples;

[Verb("run", HelpText = "Try out a trained network against your own inputs")]
public class Run : Command
{
    [Option('i', "input", HelpText = "Network input string (tensor / filepath)")]
    public IEnumerable<string>? InputStrings { get; set; }

    [Option('o', "output", HelpText = "Path to pipe output to")]
    public string? OutputPath { get; set; }

    public override void Exec(IExample example)
    {
        example.Run(InputStrings ?? Enumerable.Empty<string>(), OutputPath);
    }
}