using System.Reflection;
using CommandLine;
using DotML.Network;
using DotML.Network.IO;

namespace DotML.Examples;

public abstract class Command
{
    [Value(0, HelpText = "Example name")]
    public string? ExampleName { get; set; }

    [Option("config", HelpText = "Additional arguments to configure example behaviour in JSON form.")]
    public string? ExtraArgs { get; set; }

    private IExample GetExample()
    {
        if (string.IsNullOrEmpty(ExampleName))
            throw new NullReferenceException(nameof(ExampleName));

        var type = Assembly
            .GetExecutingAssembly()
            .GetExportedTypes()
            .Where(type => type.IsClass && !type.IsAbstract && type.IsAssignableTo(typeof(IExample)))
            .Where(type => string.Equals(type.Name, this.ExampleName, StringComparison.CurrentCultureIgnoreCase))
            .FirstOrDefault();
        if (type is null)
            throw new NotSupportedException(ExampleName);

        IExample? example = (IExample?)Activator.CreateInstance(type);
        if (example is null)
            throw new NullReferenceException(nameof(BackpropExample));

        if (!string.IsNullOrEmpty(ExtraArgs))
            example.Configure(ExtraArgs);

        return example;
    }
 
    public virtual int TryExec()
    {
        try
        {
            Exec(GetExample());
            return 0;
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            return 1;
        }
    }

    public abstract void Exec(IExample example);
}