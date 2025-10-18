using System.Reflection;
using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

public delegate Vec<float> DataLabeller(string filename, string contents);

public static class ResourceLoader {

    public static string LoadContents(string resource)
    {
        var assembly = Assembly.GetExecutingAssembly();
        //Assert.Fail(string.Join(", ",  assembly.GetManifestResourceNames()));
        var resource_str = "DotML.Test." + resource.Replace('/', '.');
        using var Stream = assembly.GetManifestResourceStream(resource_str);
        if (Stream is null)
            throw new FileNotFoundException(resource_str);

        using var Reader = new StreamReader(Stream);
        return Reader.ReadToEnd();
    }
    public static string Find(string namePart)
    {
        var assembly = Assembly.GetExecutingAssembly();
        var resources = assembly.GetManifestResourceNames();
        foreach (var name in resources)
        {
            if (!name.Contains(namePart))
                continue;

            using var Stream = assembly.GetManifestResourceStream(name);
            if (Stream is null)
                throw new FileLoadException(name);

            using var Reader = new StreamReader(Stream);
            return Reader.ReadToEnd();
        }

        throw new FileNotFoundException(namePart + " did you mean: " + string.Join(',', resources));
    }

    public static TrainingSet<float> LoadTrainingVectors(IFeatureExtractor<string, float> vectorizor, DataLabeller labeller, IEnumerable<string> resources) {
        TrainingSet<float> set = new TrainingSet<float>(
            resources.Select(
                res => {
                    var contents = LoadContents(res);
                    var label = labeller(Path.GetFileNameWithoutExtension(res), contents);
                    var data  = vectorizor.ToVector(contents);
                    
                    return new TrainingPair<float>{Input = data, Output = label};
                }
            )
        );
        return set;
    }

}