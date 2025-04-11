using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Logging;

public class Void { public static Void Instance = new Void(); private Void() {} }

public interface IOutputLogger : ILayerVisitor<(int LayerIndex, BatchedFeatureSet<double> Output), Void> { }