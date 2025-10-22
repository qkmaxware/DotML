using System.Xml.Serialization;
using DotML.Network;
using DotML.Network.IO;

namespace DotML.Cli;

public class ModelTrainingInfo {
    public string? TrainingDuration {get; set;}
    public double Accuracy {get; set;}
    public double Precision {get; set;}
    public double Recall {get; set;}
    public double MinLoss {get; set;}
    public double MaxLoss {get; set;}
    public double AvgLoss {get; set;}
}

public enum ModelTrainingStatus {
    Untrained,
    Trained
}

public class ModelProblemDescription
{
    public class ClassificationDescription
    {
        public List<string> ClassLabels { get; set; } = new List<string>();
    }
    public ClassificationDescription? Classification { get; set; }
}

public class ModelInfo
{

    private FileInfo? metadata_file = null;
    private FileInfo? network_file = null;
    private FileInfo? weights_file = null;

    public FileInfo? GetWeightsFile() => weights_file;
    public FileInfo? GetBuildScriptFile() => network_file;
    public FileInfo? GetMetadataFile() => metadata_file;

    public string? Guid => network_file is null ? string.Empty : Path.GetFileNameWithoutExtension(network_file.Name);
    public ModelTrainingStatus Status() => weights_file is null || !weights_file.Exists ? ModelTrainingStatus.Untrained : ModelTrainingStatus.Trained;
    public string? Description { get; set; }
    public ModelProblemDescription? ProblemDescription { get; set; } = new ModelProblemDescription();
    public List<string> Tags { get; set; } = new List<string>();
    public DateTime Created() => network_file is null ? DateTime.Now : network_file.CreationTime;
    public DateTime Modified() => metadata_file is null ? DateTime.Now : metadata_file.LastWriteTime;

    public ModelTrainingInfo? TrainingMetadata { get; set; }

    public ModelInfo() { }

    public ModelInfo(FileInfo metadata)
    {
        this.metadata_file = metadata;
        this.network_file = new FileInfo(Path.Combine(metadata_file.Directory?.FullName ?? string.Empty, Path.GetFileNameWithoutExtension(metadata_file.Name) + ".netbuild"));
        this.weights_file = new FileInfo(Path.Combine(metadata_file.Directory?.FullName ?? string.Empty, Path.GetFileNameWithoutExtension(metadata_file.Name) + ".safetensors"));
    }

    public ModelInfo(ModelInfo other)
    {
        if (other.metadata_file is null)
        {
            throw new ArgumentException(nameof(ModelInfo));
        }

        var guid = System.Guid.NewGuid();

        this.metadata_file = new FileInfo(Path.Combine(other.metadata_file.Directory?.FullName ?? string.Empty, guid + ".xml"));
        this.network_file = new FileInfo(Path.Combine(other.metadata_file.Directory?.FullName ?? string.Empty, guid + ".netbuild"));
        this.weights_file = new FileInfo(Path.Combine(other.metadata_file.Directory?.FullName ?? string.Empty, guid + ".safetensors"));

        if (other.network_file is not null && other.network_file.Exists)
        {
            network_file.CopyTo(network_file.FullName);
        }
        if (other.weights_file is not null && other.weights_file.Exists)
        {
            weights_file.CopyTo(weights_file.FullName);
        }
        using (var writer = new StreamWriter(metadata_file.OpenWrite()))
        {
            writer.Write(this.ToXml());
        }
    }

    public static ModelInfo? FromXml(FileInfo metadata_file)
    {
        var serializer = new XmlSerializer(typeof(ModelInfo));
        try
        {
            using (var reader = new StreamReader(metadata_file.OpenRead()))
            {
                var info = (ModelInfo?)serializer.Deserialize(reader);
                if (info is null)
                    return info;

                info.metadata_file = metadata_file;
                info.network_file = new FileInfo(Path.Combine(metadata_file.Directory?.FullName ?? string.Empty, Path.GetFileNameWithoutExtension(metadata_file.Name) + ".netbuild"));
                info.weights_file = new FileInfo(Path.Combine(metadata_file.Directory?.FullName ?? string.Empty, Path.GetFileNameWithoutExtension(metadata_file.Name) + ".safetensors"));
                return info;
            }
        }
        catch
        {
            return null;
        }
    }

    public string ToXml()
    {
        var serializer = new XmlSerializer(typeof(ModelInfo));
        using (var writer = new StringWriter())
        {
            serializer.Serialize(writer, this);
            return writer.ToString();
        }
    }

    public FeedforwardNetwork Load()
    {
        if (this.network_file is null)
            throw new FileNotFoundException();

        NetbuildSerializer builder = new NetbuildSerializer();
        var network = builder.Deserialize(File.ReadAllText(this.network_file.FullName));
        if (network is INamedNetwork named)
        {
            network.Name = Guid;
        }
        if (weights_file is not null && weights_file.Exists)
        {
            var st = Safetensors.ReadFromFile(weights_file);
            // TODO see if we need to dequantize weights
            foreach (var key in st.Keys())
            {
                var meta = st.MetadataOf(key);
                if (meta is null)
                {
                    // No metadata, skip
                    continue;
                }
                if (!meta.TryGetValue(Safetensors.QuantizationMethodKey, out var method_name))
                {
                    // No quantization method, skip
                    continue;
                }

                // Decode quantization method
                var method = decode_quantizer(method_name);
                if (method is null)
                {
                    // No quantization method, skip
                    continue;
                }

                // Apply quantization method for dequantization
                st.Dequantize(key, method);
            }
            network.FromSafetensor(st);
        }
        return network;
    }

    private IQuantization<float, byte>? decode_quantizer(string method_name)
    {
        // TODO decode quantization method
        return (method_name) switch
        {
            nameof(AbsmaxQuantization) =>
                new AbsmaxQuantization(),
            nameof(ZeroPointQuantization) =>
                new ZeroPointQuantization(),
            _ =>
                null,
        };
    }

    public void QuantizeWeights<TIn, TOut>(IQuantization<TIn, TOut> method)
    {
        if (weights_file is null || !weights_file.Exists)
            return;

        var st = Safetensors.ReadFromFile(weights_file);
        st.QuantizeAll(method);
        this.UpdateWeights(st);
    }

    public Safetensors FetchSavedWeights()
    {
        try
        {
            if (weights_file is not null && weights_file.Exists)
            {
                var st = Safetensors.ReadFromFile(weights_file);
                return st;
            }
        }
        catch { }
        return new Safetensors();
    }

    public void UpdateWeights(Safetensors tensors)
    {
        if (weights_file is null)
            return;

        using var stream = File.Open(weights_file.FullName, FileMode.Create);
        using var writer = new BinaryWriter(stream);
        tensors.WriteTo(writer);
    }

    public void UpdateWeights(FileInfo tensors)
    {
        if (weights_file is null)
            return;

        using var ostream = File.Open(weights_file.FullName, FileMode.Create);
        using var istream = tensors.OpenRead();
        istream.CopyTo(ostream);
    }

    public void UpdateBuildScript(string text)
    {
        if (network_file is null)
            return;

        using (var writer = new StreamWriter(network_file.OpenWrite()))
        {
            writer.Write(text);
        }
    }

    public string GetBuildScript()
    {
        if (network_file is null || !network_file.Exists)
            return string.Empty;

        return File.ReadAllText(network_file.FullName);
    }

    public void UpdateMetadata()
    {
        if (metadata_file is null)
            return;

        using (var writer = new StreamWriter(new FileStream(metadata_file.FullName, FileMode.Create)))
        {
            writer.Write(this.ToXml());
        }
    }

    public bool Delete()
    {
        try
        {
            if (network_file is not null && network_file.Exists)
                network_file.Delete();
            if (weights_file is not null && weights_file.Exists)
                weights_file.Delete();
            if (metadata_file is not null && metadata_file.Exists)
                metadata_file.Delete();
            return true;
        }
        catch
        {
            return false;
        }
    }
}