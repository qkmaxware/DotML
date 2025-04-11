using System.Drawing;
using CommandLine;

public class Program {

    public class Options {
        // Core options
        [Option("width", HelpText = "Image width used for output", Default = 32)]
        public int ImageWidth {get; set;}
        [Option("height", HelpText = "Image height used for output", Default = 32)]
        public int ImageHeight {get; set;}
    }

    public static void Main() {
        var args = Environment.GetCommandLineArgs().Skip(1).ToArray();
        Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(options => {
            Directory.CreateDirectory(Path.Combine("data", "images", "raw"));
            Directory.CreateDirectory(Path.Combine("data", "images", "processed"));

            var dir = new DirectoryInfo(Path.Combine("data", "images", "raw"));

            foreach (var file in dir.EnumerateFiles()) {
                using Bitmap bitmap = new Bitmap(file.FullName);
                using Bitmap scaled = Scale(bitmap, options.ImageWidth, options.ImageHeight);

                scaled.Save(Path.Combine("data", "images", "processed", file.Name));
            }
        });
    }

    public static Bitmap Scale(Bitmap original, int newWidth, int newHeight) {
        // Create a new Bitmap to hold the rotated image
        Bitmap scaledBitmap = new Bitmap(newWidth, newHeight);

        // Create a Graphics object to perform the rotation
        using (Graphics graphics = Graphics.FromImage(scaledBitmap)) {
            graphics.Clear(Color.Black);

            // Set the rendering quality for better output
            graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;

            // Scale the original image, keeping it centered within the original dimensions
            graphics.DrawImage(original, 0, 0, newWidth, newHeight);
        }

        return scaledBitmap;
    }

}