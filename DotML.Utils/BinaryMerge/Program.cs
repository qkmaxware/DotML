using System.Drawing;
using CommandLine;

namespace BinaryMerge;

public class Program {

    public class Options {
        // Core options
        [Option('f', "files", HelpText = "Files to merge", Separator = ',')]
        public IEnumerable<string>? Files {get; set;}
        [Option('o', "out", HelpText = "Output file name")]
        public string? OutName {get; set;}
    }

    public static void Main() {
        var args = Environment.GetCommandLineArgs().Skip(1).ToArray();
        Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(options => {
            Main(options);
        });
    }

    public static void Main(Options options) {
        if (string.IsNullOrEmpty(options.OutName))
            return;
        var files = options.Files;
        if (files is null || !files.Any())
            return;

        using var out_stream = File.Open(options.OutName, FileMode.Create);

        foreach (var filename in files) {
            using var in_stream = File.OpenRead(filename);
            in_stream.CopyTo(out_stream);
        }
    }

}