using System.Diagnostics.CodeAnalysis;
using System.IO;
using DotML.Network;
using DotML.Network.IO;

namespace DotML.Playground.Common.Data;

public static class Model {
    private static NetbuildSerializer netbuild = new NetbuildSerializer();
    public static bool TryLoadZip(Stream istream, [NotNullWhen(true)] out FeedforwardNetwork? network, [NotNullWhen(false)] out Exception? error) {
        network = null;
        error = null;

        try {
            using (var archive = new System.IO.Compression.ZipArchive(istream, System.IO.Compression.ZipArchiveMode.Read)) {
                var build_entry = archive.GetEntry("architecture.netbuild");
                if (build_entry is not null) {
                    using var stream = build_entry.Open();
                    using var reader = new StreamReader(stream);
                    var script = reader.ReadToEnd();
                    network = netbuild.Deserialize(script);
                }

                var meta_entry = archive.GetEntry("metadata.xml");

                var weights_entry = archive.GetEntry("weights.safetensors");
                if (weights_entry is not null && network is not null) {
                    using var wstream = weights_entry.Open();

                    using var reader = new BinaryReader(wstream);
                    var weights = Safetensors.ReadFrom(reader);
                    network.FromSafetensor(weights);
                }
            }
        } catch (Exception e) {
            error = e;
            return false;
        }

        return network is not null;
    }
}