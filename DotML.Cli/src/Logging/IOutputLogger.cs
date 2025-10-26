using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Logging;

public interface IOutputLogger
{
    public void Log(string identifier, Tensor<float> output);
}