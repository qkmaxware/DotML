using System.Reflection;
using DotML;
using DotML.Network;
using DotML.Network.Training;

namespace DotML.Sandbox.Data;

public delegate Vec<double> DataLabeller(string filename, string contents);

public static class ResourceLoader {

    public static TResource LoadJson<TResource>(string resource) => System.Text.Json.JsonSerializer.Deserialize<TResource>(LoadContents(resource)) ?? throw new FormatException($"Resource '{resource}' is not a valid JSON resource.");

    public static string LoadContents(string resource) {
        var assembly = Assembly.GetExecutingAssembly();
        //throw new Exception(string.Join(',', assembly.GetManifestResourceNames()));
        var resource_str = "DotML.Sandbox." + resource.Replace(' ', '_').Replace('/', '.');
        using var Stream = assembly.GetManifestResourceStream(resource_str);
        if (Stream is null)
            throw new FileNotFoundException(resource_str);
        
        using var Reader = new StreamReader(Stream);
        return Reader.ReadToEnd();
    }

    public static TrainingSet LoadTrainingVectors(IFeatureExtractor<string> vectorizor, DataLabeller labeller, IEnumerable<string> resources) {
        TrainingSet set = new TrainingSet(
            resources.Select(
                res => {
                    var contents = LoadContents(res);
                    var label = labeller(Path.GetFileNameWithoutExtension(res), contents);
                    var data  = vectorizor.ToVector(contents);
                    
                    return new TrainingPair{Input = data, Output = label};
                }
            )
        );
        return set;
    }

}