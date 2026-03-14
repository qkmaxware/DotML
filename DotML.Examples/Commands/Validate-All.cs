using System.Reflection;
using CommandLine;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Examples;

[Verb("test-all", HelpText = "Train all example networks (will take a long time)")]
public class ValidateAll : Command
{
    [Option("raws", HelpText = "Set flag to indicate that training data is in a raw format and needs preprocessing before training")]
    public bool ProcessRaws { get; set; }
    
    [Option("log", HelpText = "Set flag to record training logs")]
    public bool IsLogging { get; set; }

    [Option("except", HelpText = "Exclude examples from training")]
    public IEnumerable<string>? Except {get; set;} = null;

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
            .Where(instance => instance.HasBeenTrained()) // Trained only
            ;

        // User confirmation as this may take some time
        Console.WriteLine("The following networks will be trained:");
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
            if (!instance.HasBeenTrained())
                continue;

            if (ProcessRaws)
                instance.ProcessRawData();

            instance.Validate(
                useLogging: this.IsLogging
            );
        }
    }
}