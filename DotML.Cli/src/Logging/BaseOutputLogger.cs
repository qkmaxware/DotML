using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Logging;

public abstract class BaseOutputLogger : IOutputLogger {

    protected DirectoryInfo LogDirectory {get; private set;}

    public BaseOutputLogger(DirectoryInfo logDir) {
        this.LogDirectory = logDir;
        this.LogDirectory.Create();
    }

    public virtual void Log(string identifier, Tensor<float> output) {}
}