namespace DotML.Cli;

public class GenericReport {
    public DirectoryInfo Directory {get; private set;}
    public string Name => Directory.Name;

    public DateTime Created => Directory.CreationTime;
    public DateTime Modified => Directory.LastWriteTime;

    public GenericReport(DirectoryInfo info) {
        this.Directory = info;
    }

    public void Delete() {
        Directory.Delete(true);
    }
}