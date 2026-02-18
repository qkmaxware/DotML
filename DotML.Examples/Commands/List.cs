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

        Console.WriteLine("The following example projects are available.");
        Console.WriteLine("You can run an example project using the 'run' {project name} -i {input} subcommand.");
        Console.WriteLine("Examples may require training before use, see the 'train' subcommand for more information.");
        Console.WriteLine();

        WriteRow(
            (32, ("Name" + (string.IsNullOrEmpty(ExampleName) ? string.Empty : $" (filter: {ExampleName})"))),
            (16, "Problem Type"),
            (16, "Training Method"),
            (null,"Description")
        );
        foreach (var example in examples)
        {
            IExample? instance = example.GetConstructor(Type.EmptyTypes) is not null ? (Activator.CreateInstance(example) as IExample)! : null;
            
            WriteRow(
                (32, (example.Name + (instance is not null && instance.HasBeenTrained() ? " [Trained]" : string.Empty))),
                (16, ((instance is not null ? instance.Kind.ToString() : null) ?? string.Empty)),
                (16, ((instance is not null ? instance.TrainingMethod.ToString() : null) ?? string.Empty)),
                (null, ((instance is not null ? instance.GetDescription() : null) ?? string.Empty))
            );
        }
        
        Console.WriteLine();
    }

    protected void WriteRow(params ReadOnlySpan<(int? Width, object? Value)> values)
    {
        var consumedWidth = 0;
        int unsized = 0;
        foreach (var (width, value) in values)
        {
            var w = Math.Max(0, width ?? 0);
            consumedWidth += w;
            unsized += width is null ? 1 : 0;
        }

        var flexibleWidth = Math.Max(0, (Console.BufferWidth - consumedWidth) / unsized);
        foreach (var (width, value) in values)
        {
            Console.Write(ToString(value, width ?? flexibleWidth));
        }
        Console.WriteLine();
    }

    private string ToString(object? obj, int width)
    {
        var str = obj?.ToString() ?? "null";
        if (str.Length < width)
            return str.PadRight(width, ' ');
        else if (str.Length == width)
            return str;
        else
            return str.Substring(0, width);
    }
}