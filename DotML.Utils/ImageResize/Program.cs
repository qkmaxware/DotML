using System.Drawing;
using CommandLine;

namespace ImageResize;

public class Program {

    public class Options {
        // Core options
        [Option("crop", HelpText = "Crop image to aspect ratio before scaling/resizing")]
        public bool CropToAspectRatio {get; set;}
        [Option("width", HelpText = "Image width used for output", Default = 32)]
        public int ImageWidth {get; set;}
        [Option("height", HelpText = "Image height used for output", Default = 32)]
        public int ImageHeight {get; set;}
    }

    public static void Main() {
        var args = Environment.GetCommandLineArgs().Skip(1).ToArray();
        Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(options => {
            Exec(options);
        });
    }

    public static void Exec(Options options) {
        Directory.CreateDirectory(Path.Combine("data", "images", "raw"));
        Directory.CreateDirectory(Path.Combine("data", "images", "processed"));

        var dir = new DirectoryInfo(Path.Combine("data", "images", "raw"));

        foreach (var file in dir.EnumerateFiles()) {
            try {
                using Bitmap bitmap = new Bitmap(file.FullName);
                using Bitmap cropped = options.CropToAspectRatio ? Crop(bitmap, (float)bitmap.Width/bitmap.Height ) : bitmap;
                using Bitmap scaled = Scale(cropped, options.ImageWidth, options.ImageHeight);

                scaled.Save(Path.Combine("data", "images", "processed", file.Name));
            } catch { }
        }
    }

    public static Bitmap Crop(Bitmap original, float targetAspectRatio) {
        // Calculate the new dimensions based on the aspect ratio
        int originalWidth = original.Width;
        int originalHeight = original.Height;
        double originalAspectRatio = (double)originalWidth / originalHeight;

        // Calculate the crop dimensions
        int cropWidth, cropHeight;
        if (originalAspectRatio > targetAspectRatio) {
            // Image is too wide — crop width
            cropHeight = originalHeight;
            cropWidth = (int)Math.Floor(cropHeight * targetAspectRatio);
        }
        else {
            // Image is too tall — crop height
            cropWidth = originalWidth;
            cropHeight = (int)Math.Floor(cropWidth / targetAspectRatio);
        }

        // Center the crop area
        int cropX = (originalWidth - cropWidth) / 2;
        int cropY = (originalHeight - cropHeight) / 2;
        Rectangle cropRect = new Rectangle(cropX, cropY, cropWidth, cropHeight);

        Bitmap cropped = original.Clone(cropRect, original.PixelFormat);
        return cropped;
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