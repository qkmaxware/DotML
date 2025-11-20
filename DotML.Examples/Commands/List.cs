using System.Reflection;
using CommandLine;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Examples;

[Verb("list", HelpText = "List all example projects")]
public class List : Command
{

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
        var examples = Assembly
            .GetExecutingAssembly()
            .GetExportedTypes()
            .Where(type => type.IsClass && !type.IsAbstract && type.IsAssignableTo(typeof(IExample)))
            .Where(type => type.Name.Contains(this.ExampleName ?? string.Empty, StringComparison.CurrentCultureIgnoreCase))
            ;

        Console.Write("Examples: ");
        if (!string.IsNullOrEmpty(ExampleName))
        {
            Console.Write($"(filter: {ExampleName})");
        }
        Console.WriteLine();
        foreach (var example in examples)
        {
            Console.Write("  ");
            Console.Write(example.Name.PadRight(15, ' '));
            Console.Write(' ');
            // TODO description
        }
    }
}