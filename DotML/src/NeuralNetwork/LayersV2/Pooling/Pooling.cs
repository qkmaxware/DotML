using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Apply pooling to reduce the size of the image data
/// <see href="https://en.wikipedia.org/wiki/Pooling_layer"/>
/// </summary>
public abstract class Pooling : NetworkLayer
{
    // TODO maybe soem shared functionality at some point
}