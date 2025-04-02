namespace DotML.Cli;

public class TestingSummary : GenericReport {

    public TestingSummary(DirectoryInfo dir) : base(dir) { }

    public ModelInfo? ModelInfo {
        get {
            var file = Directory.EnumerateFiles().Where(file => file.Name.StartsWith("model") && file.Extension == ".xml").FirstOrDefault();
            if (file is null)
                return null;
            return ModelInfo.FromXml(file);
        }
    }

    public FileInfo? SummarySpreadsheet {
        get {
            FileInfo info = new FileInfo(Path.Combine(Directory.FullName, "summary.csv"));
            if (info.Exists)
                return info;
            return null;
        }
    }

    public FileInfo? DetailSpreadsheet { 
        get {
            FileInfo info = new FileInfo(Path.Combine(Directory.FullName, "details.csv"));
            if (info.Exists)
                return info;
            return null;
        }
    }

}