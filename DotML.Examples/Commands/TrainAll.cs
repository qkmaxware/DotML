using System.Reflection;
using CommandLine;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Examples;

[Verb("train-all", HelpText = "Test all trained networks against their validation dataset")]
public class TrainAll : Command
{
     [Option("raws", HelpText = "Set flag to indicate that training data is in a raw format and needs preprocessing before training")]
    public bool ProcessRaws { get; set; }

    [Option("pretrained", HelpText = "Set flag to indicate that saved weights should be used over random initialization")]
    public bool IsPretrained { get; set; }
    
    [Option("log", HelpText = "Set flag to record training logs")]
    public bool IsLogging { get; set; }

    [Option("save-every", HelpText = "Save the trained weights automatically every x epochs")]
    public int? SaveEvery {get; set;}

    [Option("except", HelpText = "Exclude examples from training")]
    public IEnumerable<string>? Except {get; set;} = null;

    [Option("except-trained", HelpText = "Skip training from examples that have already been trained")]
    public bool ExceptTrained {get; set;}

    public override int TryExec()
    {
        try
        {
            Exec(null);
            return 0;
        } catch
        {
            return 1;
        }
    }
    public override void Exec(IExample? _)
    {
        // Get all examples
        var examples = Assembly
            .GetExecutingAssembly()
            .GetExportedTypes()
            .Where(type => type.IsClass && !type.IsAbstract && type.IsAssignableTo(typeof(IExample)))
            .Where(type => Except is null || !Except.Contains(type.Name))
            .OrderBy(type => type.Name)
            ;
        var exampleInstances = examples
            .Select(example => example.GetConstructor(Type.EmptyTypes) is not null ? (Activator.CreateInstance(example) as IExample)! : null) // Exclude with with no default constructor
            .Where(instance => instance is not null) // Exclude nulls
            .Cast<IExample>()
            .Where(instance => !(ExceptTrained && instance.HasBeenTrained()))// Trained only
            ;

        // User confirmation as this may take some time
        Console.WriteLine("The following networks will be tested:");
        foreach (var instance in exampleInstances)
        {
            Console.Write(" - "); Console.WriteLine(instance.Name);
        }
        Console.WriteLine("This process may take a long time. Is this okay? (y/n)");
        Console.Write("> ");
        var ans = Console.ReadLine()?.ToLower();
        switch (ans)
        {
            case "y":
            case "yes":
            case "true":
            case "ja":
            case "oui":
            case "da":
            case "hai":
            case "si":
                break;
            default:
                Console.WriteLine("Training cancelled by user");
                return;
        }

        // Dispatch training
        foreach (var instance in exampleInstances)
        {
            if (ExceptTrained && instance.HasBeenTrained())
                continue;

            if (ProcessRaws)
                instance.ProcessRawData();

            instance.TrainAllVariations(
                useExistingWeights: this.IsPretrained, 
                useLogging: this.IsLogging,
                saveInterval: SaveEvery
            );
        }
    }
}