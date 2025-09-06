using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Logging;

public interface IOutputLogger : ILayerInputVisitor<(int LayerIndex, BatchedFeatureSet<float> Output)> { }