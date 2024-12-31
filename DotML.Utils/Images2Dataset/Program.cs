using System.Drawing;
using CommandLine;
using System.Text.Json;

#pragma warning disable CA1416 // Only works in Windows

public class ImagePreprocessor {

    const int IMG_WIDTH = 256;
    const int IMG_HEIGHT = 256;

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

    private static void WriteVector(byte[] vector, BinaryWriter writer) {
        writer.Write(vector.Length);
        for (var i = 0; i < vector.Length; i++) {
            writer.Write(vector[i]);
        }
    }

    private static void WriteVector(int category, byte[] vector, BinaryWriter writer) {
        writer.Write(category);                     // 4 byte category
        writer.Write(vector.Length);                // 4 bytes vector length
        for (var i = 0; i < vector.Length; i++) {
            writer.Write(vector[i]);                // Vector bytes
        }
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

    public class Options {
        // Core options
        [Option("channels", HelpText = "Channels to include in output vectors (R, G, B, RG, RB, GB, RGB, Grey).", Default = Channel.RGB)]
        public Channel Channels {get; set;}
        [Option("width", HelpText = "Image width used for scaling, resizing, and cropping. Output vectors have a size of width x height x channels.", Default = IMG_WIDTH)]
        public int ImageWidth {get; set;}
        [Option("height", HelpText = "Image height used for scaling, resizing, and cropping. Output vectors have a size of width x height x channels.", Default = IMG_HEIGHT)]
        public int ImageHeight {get; set;}

        // Un-tiling (TODO)
        /*[Option("un-tile", HelpText = "Flag to indicate if image files are tiled composites of multiple sub-images. If true images will be split into sub-images before processing. Tile size controled via the tile-width and tile-height options.", Default = false)]
        public bool IsTiled {get; set;}
        [Option("tile-width", HelpText = "Width of sub-images stored in a tiled image")]
        public int TileWidth {get; set;}
        [Option("tile-height", HelpText = "Height of sub-images stored in a tiled image")]
        public int TileHeight {get; set;}*/

        // Augmentations
        [Option("add-mirror", HelpText = "Augment the data-set by including X-Axis mirrored copied of each image", Default = false)]
        public bool AugmentMirror {get; set;}
        [Option("add-flip", HelpText = "Augment the data-set by including Y-Axis flipped copied of each image", Default = false)]
        public bool AugmentFlip {get; set;}
        [Option("add-mirrored-flip", HelpText = "Augment the data-set by including Y-Axis flipped copied of the X-axis mirror of each image", Default = false)]
        public bool AugmentMirrorFlip {get; set;}

        [Option("add-rotation", HelpText = "Augment the data-set by including rotations of each image. Rotations are provided as comma separated values in degrees.", Separator = ',', Default = null)]
        public IEnumerable<float>? AugmentRotation {get; set;}
        [Option("add-scaling", HelpText = "Augment the data-set by including scaling of each image. Scaling factors are provided as comma separated multipliers.", Separator = ',', Default = null)]
        public IEnumerable<float>? AugmentScale {get; set;}
    }

    public static void Main() {
        var args = Environment.GetCommandLineArgs().Skip(1).ToArray();
        Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(options => {
            //if (options.IsTiled && (options.TileWidth <= 0 || options.TileHeight <= 0)) {
                //throw new ArgumentException("You have indicated that images are tiled, but have provided invalid dimensions for tile-width or tile-height.");
            //}

            Directory.CreateDirectory(Path.Combine("data", "images", "raw"));
            DirectoryInfo dir = new DirectoryInfo(Path.Combine("data", "images", "raw"));
            var categories = dir.GetDirectories();
            var files_per_category = categories.Select(category => category.GetFiles()).ToArray();
            var file_count = files_per_category.Select(x => x.Length).Sum();
            //var categories_vectors = categories.Select((cat, i) => MakeVector(i, categories)).ToArray(); // [-1,-1,...1,...-1,-1]

            Directory.CreateDirectory(Path.Combine("data", "images", "processed"));
            using var binary = new BinaryWriter(File.Open(Path.Combine("data", "images", "processed", DateTime.Now.ToShortDateString() + ".vectors.bin"), FileMode.Create));
            using var labelWriter = new StreamWriter(Path.Combine("data", "images", "processed", DateTime.Now.ToShortDateString() + ".labels.csv"));

            List<Transform> transforms = [
                new Identity().Named("original"),
            ];

            if (options.AugmentMirror) {
                transforms.Add(new Flip(xflip: true, yflip: false, xyflip: false).Named("x-mirror"));
            }
            if (options.AugmentFlip) {
                transforms.Add(new Flip(xflip: false, yflip: true, xyflip: false).Named("y-flip"));
            }
            if (options.AugmentMirrorFlip) {
                transforms.Add(new Flip(xflip: false, yflip: false, xyflip: true).Named("xy-flip"));
            }

            if (options.AugmentRotation is not null && options.AugmentRotation.Any()) {
                transforms.Add(new Rotate(options.AugmentRotation.ToArray()).Named("rot"));
            }
            if (options.AugmentScale is not null && options.AugmentScale.Any()) {
                transforms.Add(new Scale(options.AugmentScale.ToArray()).Named("scaled"));
            }

            //new Scale(1.5f, 2.0f, step: 0.5f).Named("scaled"),    
            //new Rotate(-6.0f, -2.0f, step: 2.0f).Named("rneg") ,
            //new Rotate(2.0f, 6.0f, step: 2.0f).Named("rpos") 

            // Write binary header
            binary.Write(0b0001_0000);                                         // U8 As per VectorStorageType in DotML\src\NeuralNetwork\Training\TrainingData.cs
            binary.Write(1.0/255.0);                                           // Scaling from 0..255 to 0..1
            binary.Write(categories.Length);                                   // Output count
            binary.Write(file_count * transforms.Select(x => x.CreatedImageCount()).Sum()); // Input count

            // Write output vectors
            for (var i = 0; i < categories.Length; i++) {
                var vector = new byte[categories.Length];
                vector[i] = 255; // Write the max value here so it will get scaled to 1 when loaded
                WriteVector(vector, binary);
            }
            binary.Flush();

            // Write input/output vector pairs
            for (var i = 0; i < categories.Length; i++) {
                if (i > byte.MaxValue) {
                    throw new ArgumentException("Too many categories for binary encoding of images");
                }
                var categoryIndex = i;
                var category = categories[i];
                var files = files_per_category[i];
                //var output_vec = categories_vectors[i];

                Directory.CreateDirectory(Path.Combine("data", "images", "processed", category.Name));

                foreach (var file in files) {
                    Console.Write($"Processing '{file.FullName}'...");
                    using Bitmap original = new Bitmap(file.OpenRead());
                    float original_aspect = (float)original.Width / (float)original.Height;

                    foreach (var transform in transforms) {
                        // Apply perturbations to the image
                        int transform_index = 1;
                        foreach (var transformed in transform.Apply(original)) {
                            // Image has been perturbed

                            // Scale/crop image to the size of the output vector
                            using Bitmap processed = new Bitmap(options.ImageWidth, options.ImageHeight);
                            using (Graphics graphics = Graphics.FromImage(processed)) {
                                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                                graphics.Clear(Color.Black);

                                if (transformed.Width > transformed.Height) {
                                    var x_scale = (float)processed.Height/(float)transformed.Height;
                                    var x_offset = (processed.Width - (transformed.Width * x_scale)) / 2.0f;
                                    graphics.DrawImage(transformed, x_offset, 0, transformed.Width * x_scale, processed.Height);
                                } else {
                                    var y_scale = (float)processed.Width/(float)transformed.Width;
                                    var y_offset = (processed.Height - (transformed.Height * y_scale)) / 2.0f;
                                    graphics.DrawImage(transformed, 0, y_offset, processed.Width, transformed.Height * y_scale);
                                }
                            }

                            // Okay, do stuff with the final bitmap
                            var input_vec = MakeVector(processed, options.Channels);   // Create vector representation [RRRRRR, GGGGGG, BBBBBB] or whatever is selected via the channels option

                            // Save data
                            processed.Save(Path.Combine("data", "images", "processed", category.Name, Path.GetFileNameWithoutExtension(file.Name) + "." + transform.Name + "." + (transform_index++) + file.Extension));
                            WriteVector(categoryIndex, input_vec, binary);
                            binary.Flush();
                            
                            labelWriter.WriteLine(categoryIndex + ", \""+category.Name+"\"");
                            labelWriter.Flush();

                            // Cleanup images
                            processed.Dispose();
                            transformed.Dispose();
                        }
                    }

                    Console.WriteLine("done");
                }
            }

        });
    }
}