using CommandLine;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;
using System.Reflection;

namespace DotML.Examples;

[Verb("train", HelpText = "Train a given example network")]
public class Train : Command
{
    [Option("raws", HelpText = "Set flag to indicate that training data is in a raw format and needs preprocessing before training")]
    public bool ProcessRaws { get; set; }

    [Option("pretrained", HelpText = "Set flag to indicate that saved weights should be used over random initialization")]
    public bool IsPretrained { get; set; }
    
    [Option("log", HelpText = "Set flag to record training logs")]
    public bool IsLogging { get; set; }

    [Option("save-every", HelpText = "Save the trained weights automatically every x epochs")]
    public int? SaveEvery {get; set;}

    public override void Exec(IExample example)
    {
        if (ProcessRaws)
            example.ProcessRawData();
        
        example.Train(
            useExistingWeights: this.IsPretrained, 
            useLogging: this.IsLogging,
            saveInterval: SaveEvery
        );
    }
}