using System.Drawing;
using DotML;
using DotML.Network.Training;

public class Program {

    public const int InputWidth = 32; 
    public const int InputHeight = 32;
    public const int ScalingFactor = 4;

    public static void Main() {
        // Create images of input size
        if (Directory.Exists(Path.Combine("data", "images", "processed"))) {
            Directory.Delete(Path.Combine("data", "images", "processed"), true);
        }
        ImageResize.Program.Exec(new ImageResize.Program.Options {
            CropToAspectRatio = true,
            ImageWidth = InputWidth,
            ImageHeight = InputHeight,
        });
        if (Directory.Exists(Path.Combine("data", "images", "from"))) {
            Directory.Delete(Path.Combine("data", "images", "from"), true);
        }
        Directory.Move(
            sourceDirName: Path.Combine("data", "images", "processed"), 
            destDirName: Path.Combine("data", "images", "from")
        );
        // Create images of output size
        if (Directory.Exists(Path.Combine("data", "images", "processed"))) {
            Directory.Delete(Path.Combine("data", "images", "processed"), true);
        }
        ImageResize.Program.Exec(new ImageResize.Program.Options {
            ImageWidth = InputWidth * ScalingFactor,
            ImageHeight = InputHeight * ScalingFactor,
        });
        if (Directory.Exists(Path.Combine("data", "images", "to"))) {
            Directory.Delete(Path.Combine("data", "images", "to"), true);
        }
        Directory.Move(
            sourceDirName: Path.Combine("data", "images", "processed"), 
            destDirName: Path.Combine("data", "images", "to")
        );
        // Create dataset
        Images2ImagesDataset.Program.Exec(new Images2ImagesDataset.Program.Options {
            Channels = Images2ImagesDataset.Program.Channel.RGB
        });
    }

}