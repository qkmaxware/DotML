# Simple CLI Argument Parser
Very basic CLI argument parser with a minimal number of options. It is mainly focused on simple apps with a single verb\no verbs. However, it can be used with multiple verbs with a little massaging as seen in the examples below.  

## Usage
Basically one just needs to provide an ICliApp object representing WHAT the args are being parsed for, and the args themselves. 

The verb store all bare arguments (often used for things like file paths) as well as all options.

There are 3 kinds of options:
1. Option -> A basic command line option. Stores no values, but the option will have its IsSet if it exists in the arguments. `eg: -v` or `eg: --help`
2. Flag -> A slightly more complex option. Depending on if it is set or not it will return a different constant value. Useful with booleans and enumerations. `eg: --use-depreciated`
3. ValueOption&lt;T&gt; -> An option that can store an associated value. The value must be able to be parsed from a string representation. This can be done by ensuring that the type T extends from IParsable. `eg: --api-version 12`

Once parsed, you can access the unparsed\bare arguments of the ICliApp and can use it's options to extract if flags exist or what values the options contain. This code all depends on you though and is outside of this library's scope. 

## Examples
Single\No Verb APP
```cs
using Qkmaxware.Simple.CommandLineParser;

public class Program: ICliApp {

    private List<string> args = new();
    public void AddArgument(string str) => args.Add(str);
    
    private List<Option> opts = new List<Option> {
        new Option(null, "help", "Get app help")
    };
    public IEnumerable<Option> GetOptions() => opts;

    public static void Main() {
        var parser = new Parser();
        var program = new Program();
        parser.ParseCommandLineArgs(program);
        program.Run();
    }

    void Run() {
        if (opts.First(o => o.LongName == "help").IsSet) {
            // Print help here
            return;
        }

        // Do work here
    }
}
```

Multiple Verb APPS (might clean this up one of these days)
```cs
using Qkmaxware.Simple.CommandLineParser;

public class Verb0: ICliApp {
    // options and such ...
}

public class Verb1: ICliApp {
    // options and such ...
}

public class Program {
    public static void Main() {
        var _0 = new Verb0();
        var _1 = new Verb1();

        var args = System.Environment.GetCommandLineArgs();
        var appName = args[0];
        var verbName = args[1];

        var verb = verbName switch {
            "firstVerbName" => _0,
            "secondVerbName" => _1,
            _ => throw new Exception("Unknown verb " + verbName);
        };
        
        var parser = new Parser();
        parser.ParseArgs(verb, args.Skip(2), skipFirst: false);
        
        // Do whatever with the verb and it's options/bare args
    }
}
```