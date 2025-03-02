namespace DotML.Cli;

public class TrainingReport {

    private DirectoryInfo dir;

    public string Name => dir.Name;

    public TrainingReport(DirectoryInfo dir) {
        this.dir = dir;
    }

    public ModelInfo? ModelInfo {
        get {
            var file = dir.EnumerateFiles().Where(file => file.Name.StartsWith("model") && file.Extension == ".xml").FirstOrDefault();
            if (file is null)
                return null;
            return ModelInfo.FromXml(file);
        }
    }

    public FileInfo? TrainerConfig {
        get {
            FileInfo info = new FileInfo(Path.Combine(dir.FullName, "trainer-config.yaml"));
            if (info.Exists)
                return info;
            return null;
        }
    }

    public FileInfo? NetworkDescription {
        get {
            FileInfo info = new FileInfo(Path.Combine(dir.FullName, "network-description.md"));
            if (info.Exists)
                return info;
            return null;
        }
    }

    public FileInfo? ValidationChart {
        get {
            FileInfo info = new FileInfo(Path.Combine(dir.FullName, "validation.csv"));
            if (info.Exists)
                return info;
            return null;
        }
    }

    public FileInfo? TestingChart {
        get {
            FileInfo info = new FileInfo(Path.Combine(dir.FullName, "testing.csv"));
            if (info.Exists)
                return info;
            return null;
        }
    }

    public IEnumerable<FileInfo> RetainedWeights {
        get {
            return new DirectoryInfo(Path.Combine(dir.FullName, "weights")).EnumerateFiles("*.safetensors");
        }
    }
}