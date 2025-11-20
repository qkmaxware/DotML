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
        // Load network
        var network = example.GetArchitecture();

        // Load weights (required)
        RestoreWeights(example, network, throws: true);

        // Parse user input
        if (InputStrings is null)
            return;

        using TextWriter pipe = !string.IsNullOrEmpty(OutputPath) ? CreateLogger("output.txt") : System.Console.Out;
        foreach (var str in InputStrings)
        {
            pipe.Write("> "); pipe.WriteLine(str);
            var input = example.ParseUserInput(str);
            var output = network.Forward(input);
            pipe.WriteLine(example.FormatOutput(str, input, output));
            pipe.WriteLine(); // Extra line between inputs
        }
    }
}