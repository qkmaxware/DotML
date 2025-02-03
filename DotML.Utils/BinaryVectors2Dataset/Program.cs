using System.Drawing;
using CommandLine;
using DotML;
using DotML.Network.Training;

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
            if (string.IsNullOrEmpty(options.OutName))
                return;
            var files = options.Files;
            if (files is null || !files.Any())
                return;
            options.OutName = options.OutName.EndsWith(".bin") ? options.OutName : options.OutName + ".bin";

            var parser = new BinaryClassifiedVectors();
            TrainingSet set = new TrainingSet();
            foreach (var file in files) {
                var info = new FileInfo(file);
                if (!info.Exists)
                    continue;
                set.AddRange(parser.Read(info));
            }

            using var out_stream = File.Open(options.OutName, FileMode.Create);
            using var writer = new BinaryWriter(out_stream);
            set.WriteTo(writer);
            Console.WriteLine("Training set contains: " + set.Size + " entries");
            Console.WriteLine("File saved as " + options.OutName);
        });
    }

}

public class BinaryClassifiedVectors {

    public int OutputClasses = 10;
    public double ZeroValue = 0.0;
    public double OneValue = 1.0;

    public IEnumerable<TrainingPair> Read(FileInfo file) {
        return read_classified_binary_vectors(
            file:               file, 
            category_off:       ZeroValue, 
            category_on:        OneValue, 
            element_parser:     x => (x.ReadByte() / 255.0), 
            fixed_vector_size:  1024 * 3 
        );
    }

    private static Vec<double> vector_from_label_index(int index, int classes, double off = -1, double on = 1) {
        double[] values = new double[classes];
        Array.Fill(values, off);
        if (index >= 0 && index < classes)
            values[index] = on;
        return Vec<double>.Wrap(values);
    }

    private static IEnumerable<TrainingPair> read_classified_binary_vectors(FileInfo file, double category_off, double category_on, Func<BinaryReader, double> element_parser, int? fixed_vector_size = null) {
        using var stream = file.OpenRead();
        using var reader = new BinaryReader(stream);
        
        List<(Vec<double>, int)> items = new List<(Vec<double>, int)>();
        int category_count = 1;
        while (stream.Position < stream.Length) {
            var category_index  = reader.ReadByte();
            category_count = Math.Max(category_count, category_index + 1);
            var vector_size     = fixed_vector_size.HasValue ? fixed_vector_size.Value : reader.ReadInt32();
            double[] input_vec  = new double[vector_size];

            for (var i = 0; i < vector_size; i++) {
                try {
                    input_vec[i] = element_parser(reader);
                } catch {
                    input_vec[i] = default(double);
                }
            } 
            items.Add((Vec<double>.Wrap(input_vec), category_index));
        }
        
        return items.Select(item => new TrainingPair { Input=item.Item1, Output=vector_from_label_index(item.Item2, category_count, category_off, category_on) });
    }
}