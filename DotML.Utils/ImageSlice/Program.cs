using System.Drawing;
using CommandLine;

#pragma warning disable CA1416

public class Program {

    const int TILE_WIDTH = 32;
    const int TILE_HEIGHT = 32;

    public class Options {
        // Core options
        [Option("width", HelpText = "Image width used for scaling, resizing, and cropping. Output vectors have a size of width x height x channels.", Default = TILE_WIDTH)]
        public int ImageWidth {get; set;}
        [Option("height", HelpText = "Image height used for scaling, resizing, and cropping. Output vectors have a size of width x height x channels.", Default = TILE_HEIGHT)]
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

                int slice_index = 0;
                foreach (var submap in Slice(bitmap, options.ImageWidth, options.ImageHeight)) {
                    Directory.CreateDirectory(Path.Combine("data", "images", "processed", slice_index.ToString()));
                    submap.Save(Path.Combine("data", "images", "processed", slice_index.ToString(), file.Name));

                    // Done
                    slice_index++;
                    submap.Dispose();
                }
            }

        });
    }

    private static IEnumerable<Bitmap> Slice(Bitmap original, int tileWidth, int tileHeight) {
        // Get the width and height of the original image
        int imageWidth = original.Width;
        int imageHeight = original.Height;

        // Calculate the number of tiles needed in both dimensions
        int columns = (int)Math.Ceiling((double)imageWidth / tileWidth);
        int rows = (int)Math.Ceiling((double)imageHeight / tileHeight);

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < columns; x++) {
                // Calculate the rectangle for each tile
                int xPos = x * tileWidth;
                int yPos = y * tileHeight;

                // Ensure the rectangle does not go out of bounds
                int width = Math.Min(tileWidth, imageWidth - xPos);
                int height = Math.Min(tileHeight, imageHeight - yPos);

                // Crop the image and create a new Bitmap for each tile
                Rectangle cropRect = new Rectangle(xPos, yPos, width, height);
                Bitmap tile = original.Clone(cropRect, original.PixelFormat);

                yield return tile;
            }
        }
    }
}