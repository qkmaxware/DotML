namespace DotML.Cli;

public class TrainingSession : GenericReport {

    public TrainingSession(DirectoryInfo dir) : base(dir) { }

    public ModelInfo? ModelInfo {
        get {
            var file = Directory.EnumerateFiles().Where(file => file.Name.StartsWith("model") && file.Extension == ".xml").FirstOrDefault();
            if (file is null)
                return null;
            return ModelInfo.FromXml(file);
        }
    }

    public FileInfo? TrainerConfig {
        get {
            FileInfo info = new FileInfo(Path.Combine(Directory.FullName, "trainer-config.yaml"));
            if (info.Exists)
                return info;
            return null;
        }
    }

    public FileInfo? NetworkDescription {
        get {
            FileInfo info = new FileInfo(Path.Combine(Directory.FullName, "network-description.md"));
            if (info.Exists)
                return info;
            return null;
        }
    }

    public FileInfo? ValidationChart {
        get {
            FileInfo info = new FileInfo(Path.Combine(Directory.FullName, "validation.csv"));
            if (info.Exists)
                return info;
            return null;
        }
    }

    public FileInfo? TestingChart {
        get {
            FileInfo info = new FileInfo(Path.Combine(Directory.FullName, "testing.csv"));
            if (info.Exists)
                return info;
            return null;
        }
    }

    public IEnumerable<FileInfo> RetainedWeights {
        get {
            return new DirectoryInfo(Path.Combine(Directory.FullName, "weights")).EnumerateFiles("*.safetensors");
        }
    }
}