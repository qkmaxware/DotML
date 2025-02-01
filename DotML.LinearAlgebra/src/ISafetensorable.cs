using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace DotML;

/// <summary>
/// Any object that is serializable to safetensor format
/// </summary>
public interface ISafetensorable {
    /// <summary>
    /// Output this network's configuration in the safetensor format
    /// </summary>
    /// <param name="writer">binary writer to write to</param>
    public void ToSafetensor(BinaryWriter writer);

    /// <summary>
    /// Output this network's configuration in the safetensor format
    /// </summary>
    /// <returns>safetensor bytes</returns>
    public byte[] ToSafetensor() {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        ToSafetensor(writer);

        return stream.ToArray();
    }
}