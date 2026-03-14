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

public class ArchitectureFile
{
    private string dir;
    private string nameGuid;
    private FileInfo[] potentialFiles;
    public ArchitectureFile(string dir, string nameGuid)
    {
        this.nameGuid = nameGuid;
        this.dir = dir;
        potentialFiles = ModuleParser.AllowedExtensions.Select(ext => new FileInfo(Path.Combine(nameGuid + ext))).ToArray();
    }

    public string Extension
    {
        get
        {
            foreach (var file in potentialFiles)
            {
                if (file.Exists)
                    return file.Extension;
            }
            return ModuleParser.PreferredExtension;
        }
    }

    public string Name
    {
        get
        {
            foreach (var file in potentialFiles)
            {
                if (file.Exists)
                {
                    return file.Name;
                }
            }

            return nameGuid + ModuleParser.PreferredExtension;
        }
    }

    public string FullName
    {
        get
        {
            foreach (var file in potentialFiles)
            {
                if (file.Exists)
                {
                    return file.FullName;
                }
            }

            return Path.Combine(dir, nameGuid + ModuleParser.PreferredExtension);
        }
    }

    public bool Exists
    {
        get
        {
            foreach (var file in potentialFiles)
            {
                if (file.Exists)
                {
                    return true;
                }
            }
            return false;
        }
    }

    public DateTime CreationTime
    {
        get
        {
            return potentialFiles.Where(file => file.Exists).Select(file => file.CreationTime).Min();
        }
    }

    public DateTime LastWriteTime
    {
        get
        {
            return potentialFiles.Where(file => file.Exists).Select(file => file.LastWriteTime).Max();
        }
    }

    public void Delete()
    {
        foreach (var file in potentialFiles)
        {
            if (file.Exists)
                file.Delete();
        }
    }

    public void CopyTo(string path)
    {
        var pathwithoutExt = Path.GetFileNameWithoutExtension(path);
        foreach (var file in this.potentialFiles)
        {
            // Copy first existing file, preserve extension regardless of what the path says
            if (file.Exists)
            {
                file.CopyTo(pathwithoutExt + file.Extension);
                return;
            }
        }
    }

    public string ReadAllText()
    {
        foreach (var file in potentialFiles)
        {
            if (file.Exists)
                return File.ReadAllText(file.FullName);
        }
        return string.Empty;
    }
}

public class ModelInfo
{

    private FileInfo? metadata_file = null;
    private FileInfo? weights_file = null;

    private ArchitectureFile? architecture_file = null;

    public FileInfo? GetWeightsFile() => weights_file;
    public ArchitectureFile? GetBuildScriptFile() => architecture_file;
    public FileInfo? GetMetadataFile() => metadata_file;

    public string? Guid => metadata_file is null ? string.Empty : Path.GetFileNameWithoutExtension(metadata_file.Name);
    public ModelTrainingStatus Status() => weights_file is null || !weights_file.Exists ? ModelTrainingStatus.Untrained : ModelTrainingStatus.Trained;
    public string? Description { get; set; }
    public ModelProblemDescription? ProblemDescription { get; set; } = new ModelProblemDescription();
    public List<string> Tags { get; set; } = new List<string>();
    public DateTime Created() => architecture_file is null ? DateTime.Now : architecture_file.CreationTime;
    public DateTime Modified() => metadata_file is null ? DateTime.Now : metadata_file.LastWriteTime;

    public ModelTrainingInfo? TrainingMetadata { get; set; }

    public ModelInfo() { }

    public ModelInfo(FileInfo metadata)
    {
        this.metadata_file = metadata;
        this.architecture_file = new ArchitectureFile(metadata_file.Directory?.FullName ?? string.Empty, Path.GetFileNameWithoutExtension(metadata_file.Name));
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
        this.architecture_file = new ArchitectureFile(other.metadata_file.Directory?.FullName ?? string.Empty, guid.ToString());
        this.weights_file = new FileInfo(Path.Combine(other.metadata_file.Directory?.FullName ?? string.Empty, guid + ".safetensors"));

        if (other.architecture_file is not null && other.architecture_file.Exists)
        {
            other.architecture_file.CopyTo(architecture_file.FullName);
        }
        if (other.weights_file is not null && other.weights_file.Exists)
        {
            other.weights_file.CopyTo(weights_file.FullName);
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
                info.architecture_file = new ArchitectureFile(metadata_file.Directory?.FullName ?? string.Empty, Path.GetFileNameWithoutExtension(metadata_file.Name));
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

    public INetworkModule Load()
    {
        if (this.architecture_file is null || !this.architecture_file.Exists)
            throw new FileNotFoundException();

        var module = ModuleParser.Parse(this.architecture_file.ReadAllText(), this.architecture_file.Extension);

        // Name the module by wrapping it with a architecture block
        module = new ArchitectureBlock(this.Guid ?? "Network", null, module);

        // Load weights
        ReloadWeights(module);

        return module;
    }
    
    public void ReloadWeights(INetworkModule module)
    {
        var loader = new SafetensorDeserializer();
        if (weights_file is not null && weights_file.Exists)
        {
            var st = Safetensors.ReadFromFile(weights_file);

            // Dequantize weights if quantized
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

            // Apply the weights
            if (module is IBlockVisitable visitableModule)
                visitableModule.Accept(loader, st);
        }
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

    public string GetBuildScript()
    {
        if (architecture_file is null || !architecture_file.Exists)
            return string.Empty;

        return architecture_file.ReadAllText();
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
            if (architecture_file is not null && architecture_file.Exists)
                architecture_file.Delete();
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