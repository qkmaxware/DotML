namespace Qkmaxware.Simple.CommandLineParser;

public class Option
{
    public char? ShortName;
    public string LongName;

    public string? Description;

    public Option(char? shortName, string longName)
    {
        this.ShortName = shortName;
        this.LongName = longName;
    }

    public bool IsSet {get; set;} = false;
}

public class Flag<T> : Option
{
    private T setValue;
    private T unsetValue;

    public T Value => IsSet ? setValue : unsetValue;

    public Flag(char? shortName, string longName, T setValue, T unsetValue) : base(shortName, longName)
    {
        this.setValue = setValue;
        this.unsetValue = unsetValue;
    }
}

public abstract class ValueOption : Option
{
    protected ValueOption(char? shortName, string longName) : base(shortName, longName)
    {
    }

    public abstract object? GetValue();
    public abstract bool TrySetValue(object? obj);
    public abstract bool TryParseValue(string? str);
}

public class ValueOption<T> : ValueOption
where T : IParsable<T>
{
    public ValueOption(char? shortName, string longName) : base(shortName, longName)
    {
    }

    private T? Value {get; set;}

    public override object? GetValue() => Value;

    public override bool TryParseValue(string? str)
    {
        if (T.TryParse(str, null, out T? result))
        {
            this.Value = result;
            return true;
        }
        
        return false;
    }

    public override bool TrySetValue(object? obj)
    {
        if (obj is null)
        {
            this.Value = default(T);
            return true;
        }

        if (obj is T value)
        {
            this.Value = value;
            return true;
        }

        return false;
    }
}