namespace Qkmaxware.Simple.CommandLineParser;

public class Parser
{
    public void ParseCommandLineArgs(ICliApp verb) => ParseArgs(verb, System.Environment.GetCommandLineArgs(), skipFirst: true);
    public void ParseArgs(ICliApp verb, IEnumerable<string> args, bool skipFirst = true)
    {
        var enumerator = args.GetEnumerator();

        if (skipFirst)
        {
            enumerator.MoveNext();
        }

        while (enumerator.MoveNext())
        {
            var arg = enumerator.Current;

            if (arg.StartsWith("--"))
            {
                Option? opt = verb.GetOptions().FirstOrDefault((o) => MemoryExtensions.Equals(arg.AsSpan(2), o.LongName, StringComparison.CurrentCultureIgnoreCase));
                if (opt is null)
                    throw new ArgumentException($"Unknown option '{arg}'.");
                opt.IsSet = true;
                if (opt is ValueOption vOpt) {
                    var optValue = enumerator.MoveNext() ? enumerator.Current : string.Empty;
                    vOpt.TryParseValue(optValue);
                }
            } else if (arg.StartsWith('-'))
            {
                char c = arg.Length > 1 ? arg[1] : '\0';
                Option? opt = verb.GetOptions().FirstOrDefault((o) => o.ShortName.HasValue && o.ShortName == c);
                if (opt is null)
                    throw new ArgumentException($"Unknown option '{arg}'.");
                opt.IsSet = true;
                if (opt is ValueOption vOpt) {
                    var optValue = enumerator.MoveNext() ? enumerator.Current : string.Empty;
                    vOpt.TryParseValue(optValue);
                }
            } else
            {
                if (arg is not null)    
                    verb.AddArgument(arg);
            }
        }
    }
}
