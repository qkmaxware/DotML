using System.Reflection;
using DotML;
using DotML.Network;
using DotML.Network.Training;

namespace DotML.Sandbox.Data;

public delegate Vec<double> DataLabeller(string filename, string contents);

public static class ResourceLoader {

    public static TResource LoadJson<TResource>(string resource) => System.Text.Json.JsonSerializer.Deserialize<TResource>(LoadContents(resource)) ?? throw new FormatException($"Resource '{resource}' is not a valid JSON resource.");

    public static Stream GetResourceStream(string resource) {
        var assembly = Assembly.GetExecutingAssembly();
        var resource_str = "DotML.Sandbox." + resource.Replace(' ', '_').Replace('/', '.');
        var Stream = assembly.GetManifestResourceStream(resource_str);
        if (Stream is null)
            throw new FileNotFoundException(resource_str);
        return Stream;
    }

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

    public static SafetensorBuilder LoadSafetensors(string resource) {
        using var Reader = new BinaryReader(GetResourceStream(resource));
        return SafetensorBuilder.ReadFrom(Reader);
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

    private static Vec<double> VectorFromLabelIndex(int index, int classes, double off = -1, double on = 1) {
        double[] values = new double[classes];
        Array.Fill(values, off);
        if (index >= 0 && index < classes)
            values[index] = on;
        return Vec<double>.Wrap(values);
    }
    public static TrainingSet LoadBinaryVectors(IEnumerable<string> resources, int category_count, double category_off, double category_on, Func<BinaryReader, double> element_parser) {
        TrainingSet set = new TrainingSet(
            resources.SelectMany(
                res => {
                    using var stream = GetResourceStream(res);
                    using var reader = new BinaryReader(stream);
                    
                    List<TrainingPair> pairs = new List<TrainingPair>();
                    while (stream.Position < stream.Length) {
                        var category_index  = reader.ReadByte();
                        var vector_size     = reader.ReadInt32();
                        double[] input_vec  = new double[vector_size];

                        for (var i = 0; i < vector_size; i++) {
                            try {
                                input_vec[i] = element_parser(reader);
                            } catch {
                                input_vec[i] = default(double);
                            }
                        } 
                        pairs.Add(new TrainingPair { Input = Vec<double>.Wrap(input_vec), Output = VectorFromLabelIndex(category_index, category_count, category_off, category_on) });
                    }

                    return pairs;
                }
            )
        );
        return set;
    }

}