using System.Drawing;
using CommandLine;
using DotML.Network.Training;
using DotML;

public class Program {

    public class Options {
        // Core options
        [Option("channels", HelpText = "Channels to include in output vectors (R, G, B, RG, RB, GB, RGB, Grey).", Default = Channel.RGB)]
        public Channel Channels {get; set;}
    }

    public enum Channel {
        R, G, B, RG, RB, GB, RGB, Grey
    }
    public static int ChannelCount(Channel channel) {
        return channel switch {
            Channel.RGB => 3,
            Channel.RG => 2,
            Channel.RB => 2,
            Channel.GB => 2,
            _ => 1,
        };
    }

    private static byte[] MakeVector(Bitmap bmp, Channel channels) {
        var frame_size = bmp.Width * bmp.Height;
        var vector = new byte[frame_size * ChannelCount(channels)]; // RGB
        for (var row = 0; row < bmp.Height; row++) {
            for (var col = 0; col < bmp.Width; col++) {
                var colour = bmp.GetPixel(col, row);
                switch (channels) {
                    case Channel.Grey:
                        var greyscale = (0.299 * colour.R + 0.587 * colour.G + 0.114 * colour.B);
                        vector[row * bmp.Width + col + 0*frame_size] = (byte)Math.Clamp(greyscale, 0, 255);
                        break;
                    case Channel.RGB:
                        vector[row * bmp.Width + col + 0*frame_size] = colour.R;
                        vector[row * bmp.Width + col + 1*frame_size] = colour.G;
                        vector[row * bmp.Width + col + 2*frame_size] = colour.B;
                        break;
                    case Channel.GB:
                        vector[row * bmp.Width + col + 0*frame_size] = colour.G;
                        vector[row * bmp.Width + col + 1*frame_size] = colour.B;
                        break;
                    case Channel.RB:
                        vector[row * bmp.Width + col + 0*frame_size] = colour.R;
                        vector[row * bmp.Width + col + 1*frame_size] = colour.B;
                        break;
                    case Channel.RG:
                        vector[row * bmp.Width + col + 0*frame_size] = colour.R;
                        vector[row * bmp.Width + col + 1*frame_size] = colour.G;
                        break;
                    case Channel.R:
                        vector[row * bmp.Width + col + 0*frame_size] = colour.R;
                        break;
                    case Channel.G:
                        vector[row * bmp.Width + col + 0*frame_size] = colour.G;
                        break;
                    case Channel.B:
                        vector[row * bmp.Width + col + 0*frame_size] = colour.B;
                        break;
                }
            }
        }
        return vector;
    }

    public static void Main() {
        var args = Environment.GetCommandLineArgs().Skip(1).ToArray();
        Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(options => {
            Directory.CreateDirectory(Path.Combine("data", "images", "from"));
            Directory.CreateDirectory(Path.Combine("data", "images", "to"));
            Directory.CreateDirectory(Path.Combine("data", "datasets"));

            var from = new DirectoryInfo(Path.Combine("data", "images", "from"));
            var to = new DirectoryInfo(Path.Combine("data", "images", "to"));

            TrainingSet data = new TrainingSet();
            foreach (var file in from.EnumerateFiles()) {
                var pair = Path.Combine("data", "images", "to", file.Name);
                if (!File.Exists(pair))
                    continue;

                using Bitmap original = new Bitmap(file.FullName);
                using Bitmap modified = new Bitmap(pair);
                
                var original_bytes = MakeVector(original, options.Channels);
                var scaled_original_bytes = original_bytes.Select(x => x / 255.0).ToArray();

                var modified_bytes = MakeVector(modified, options.Channels);
                var scaled_modified_bytes = modified_bytes.Select(x => x / 255.0).ToArray();

                data.Add(new TrainingPair { Input = Vec<double>.Wrap(scaled_original_bytes), Output = Vec<double>.Wrap(scaled_modified_bytes) });
            }

            using var writer = new BinaryWriter(File.OpenWrite(Path.Combine("data", "datasets", "training.bin")));
            data.WriteTo(writer);
        });
    }

}