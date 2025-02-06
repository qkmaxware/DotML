using DotML.Network;

namespace DotML.Cli;

public class AppData {

    private string root_dir;
    private string training_dir;
    private string model_dir;

    public AppData() {
        var home = Environment.GetEnvironmentVariable("NETFLOW_HOME");
        var app_data = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "DotML.NetFlow");

        this.root_dir = home ?? app_data;
        Directory.CreateDirectory(root_dir);

        this.training_dir = Path.Combine(root_dir, "Training");
        Directory.CreateDirectory(training_dir);

        this.model_dir = Path.Combine(root_dir, "Models");
        Directory.CreateDirectory(model_dir);
    }

    public DirectoryInfo ModelDirectory => new DirectoryInfo(this.model_dir);

    public IEnumerable<ModelInfo> ListModels() {
        foreach (var file in ModelDirectory.GetFiles("*.xml")) {
            var info = ModelInfo.FromXml(file);
            if (info is not null)
                yield return info;
        }
    }

    public ModelInfo? GetModel(string? name) {
        return ListModels().Where(
            model => 
                (model.Guid is null ? false : model.Guid.Equals(name, StringComparison.CurrentCultureIgnoreCase))
                || (name is null ? false : model.Tags.Where(tag => tag.Contains(name, StringComparison.CurrentCultureIgnoreCase)).Any())
        ).FirstOrDefault();
    }

    public DirectoryInfo CreateTrainingDir() {
        var now = DateTime.Now.ToString("yyyy-dd-M--HH-mm-ss");
        var path = Path.Combine(training_dir, now);
        Directory.CreateDirectory(path);
        return new DirectoryInfo(path);
    }
}