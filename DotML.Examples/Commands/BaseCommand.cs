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
            throw new NullReferenceException(nameof(Example));

        if (!string.IsNullOrEmpty(ExtraArgs))
            example.Configure(ExtraArgs);

        return example;
    }

    private DateTime startTime = DateTime.Now;

    protected StreamWriter CreateLogger(string name)
    {
        return new StreamWriter(startTime.ToString("yyyy-dd-M--HH-mm-ss") + "." + name);
    }

    protected void RestoreWeights(IExample example, INetworkModule network, bool throws = false)
    {
        try
        {
            var tensors = example.LoadWeights();
            var applier = new SafetensorDeserializer();
            if (network is IBlockVisitable visitable)
                applier.Deserialize(visitable, tensors);
        }
        catch (Exception)
        {
            // Re-throw if configured to
            if (throws)
                throw;
        }
    }

    protected void SaveWeights(IExample example, INetworkModule network, bool throws = false)
    {
        try
        {
            var applier = new SafetensorSerializer();
            if (network is IBlockVisitable visitable)
                applier.Serialize(visitable);
            example.SaveWeights(applier.ToSafetensors());
        }
        catch (Exception)
        {
            // Re-throw if configured to
            if (throws)
                throw;
        }
    }

    protected void WriteRow(params ReadOnlySpan<object?> values)
    {
        var columnWidth = Console.BufferWidth / values.Length;
        for (var i = 0; i < values.Length; i++)
        {
            Console.Write(ToString(values[i], columnWidth));
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