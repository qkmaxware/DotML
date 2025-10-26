using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Logging;

public class TensorsOutputLogger : BaseOutputLogger {

    public TensorsOutputLogger(DirectoryInfo logDir) : base(logDir) { }

    public override void Log(string identifier, Tensor<float> output)
    {
        var dir_path = Path.Combine(LogDirectory.FullName, identifier);
        var dir = Directory.CreateDirectory(dir_path);

        using var writer = new StreamWriter(Path.Combine(dir.FullName, "output.tensor.xml"));
        output.SaveSpreadsheetML(writer);
    }
}