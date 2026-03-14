using System.Drawing;
using DotML;
using DotML.Network.Training;

public class Program {

    public static int MasksPerImage = 4;
    public static int InstancesPerMask = 2;
    public static int MinMaskSize = 4;
    public static int MaxMaskSize = 12;

    public static void Main() {
        var files = new DirectoryInfo(Path.Combine(Environment.CurrentDirectory, "..", "Cifar-10", "raw")).GetFiles("*.bin");
        var builder = new BinaryVectorBuilder<byte>();
        builder.ScalingFactor = 1.0 / 255.0;
        foreach (var file in files) {
            Console.Write($"Converting '{file.Name}'...");
            builder.AddRange(ReadBytes(file, 32, 32, 3));
            Console.WriteLine("done");
        }
        Console.WriteLine();
        Console.WriteLine("Total outputs: " + builder.Count / MasksPerImage);
        Console.WriteLine("Total inputs: " + builder.Count);
        
        Console.WriteLine();
        var output_path = "training.bin";
        Console.Write($"Writing training set '{output_path}'...");
        using var stream = File.Open("training.bin", FileMode.Create);
        using var writer = new BinaryWriter(stream);
        builder.WriteTo(writer);
        Console.WriteLine("done");
    }

    /// <summary>
    /// Colour channel enumeration
    /// </summary>
    public enum Channel {
        /// <summary>
        /// Red channel '0'
        /// </summary>
        R = 0,
        /// <summary>
        /// Green channel '1'
        /// </summary>
        G = 1, 
        /// <summary>
        /// Blue channel '2'
        /// </summary>
        B = 2
    }

    private struct Img2D<T> {
        public int Height {get; private set;}
        public int Width {get; private set;}
        public int Channels {get; private set;}
        private T[] data;
        public Img2D(int rows, int cols, int channels, T[] data) {
            this.Height = rows;
            this.Width = cols;
            this.Channels = channels;
            this.data = data;
        } 

        public T this[Channel channel, int x, int y] {
            get => this[(int)channel, x, y];
            set => this[(int)channel, x, y] = value;
        }
        public T this[int channel, int x, int y] {
            get => data[Height*Width*((int)channel) + y * Width + x];
            set {
                if (channel >= 0 && channel < Channels && x >= 0 && x < Width && y >= 0 && y < Height) {
                    data[Height*Width*((int)channel) + y * Width + x] = value;
                }
            }
        }

        public Img2D<T> Clone() {
            T[] output_vec  = new T[data.Length];
            data.AsSpan().CopyTo(output_vec);
            return new Img2D<T>(this.Height, this.Width, this.Channels, output_vec);
        }

        public T[] AsArray() => data;
    }

    private static Random rng = new Random();

    public static IEnumerable<KeyValuePair<byte[], byte[]>> ReadBytes(FileInfo file, int width, int height, int channels, bool has_category = true) {
        using var stream = file.OpenRead();
        using var reader = new BinaryReader(stream);
        
        var size = width*height*channels;

        while (stream.Position < stream.Length) {
            if (has_category) {
                var category = reader.ReadByte(); // IGNORE
            }

            var vector_size     = size;
            byte[] input_vec  = new byte[vector_size];
            for (var i = 0; i < vector_size; i++) {
                try {
                    input_vec[i] = reader.ReadByte();
                } catch {
                    input_vec[i] = (byte)0;
                }
            } 
            var input_image = new Img2D<byte>(height, width, channels, input_vec);
            
            // TODO do a bunch of masking to the input image to create different "infill" masks
            for (var i = 0; i < MasksPerImage; i++) {
                var edited_input = input_image.Clone();

                var mask_count = InstancesPerMask;
                for (var j = 0; j < mask_count; j++) { 
                    var mask_size = rng.Next(minValue: MinMaskSize, maxValue: MaxMaskSize + 1);
                    var mask_half_size = mask_size / 2;
                    var mask_center = new Point(x: mask_half_size + rng.Next(edited_input.Width - mask_size), y: mask_half_size + rng.Next(edited_input.Height - mask_size));
                    for (int y = mask_center.Y - mask_half_size, my = 0; my < mask_size; my++, y++) { 
                        for (int x = mask_center.X - mask_half_size, mx = 0; mx < mask_size; mx++, x++) {
                            // Clear the pixel
                            edited_input[Channel.R, x, y] = 0;
                            edited_input[Channel.G, x, y] = 0;
                            edited_input[Channel.B, x, y] = 0;
                        }
                    }
                }

                yield return new KeyValuePair<byte[], byte[]>(edited_input.AsArray(), input_image.AsArray());
            }
        }
    }
}