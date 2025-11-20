using DotML.Network;
using DotML.Network.Training;

namespace DotML.Examples;

public interface IExample
{
    public string Name { get; }
    public void Configure(string json);

    public void Clean();

    public INetworkModule GetArchitecture();
    public Safetensors LoadWeights();
    public void SaveWeights(Safetensors tensors);

    public Tensor<float> ParseUserInput(string input);
    public void ProcessRawData();
    public void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation);
    public ITrainingDataSampler<float> GetTrainingSampler(ITrainingDataSource<float> training);
    public ITrainingDataSampler<float> GetValidationSampler(ITrainingDataSource<float> validation);
    public void ConfigureTrainer(ModuleTrainer trainer);
    public IEnumerable<IReport> GenerateTrainingReports(INetworkModule network, ITrainingDataSource<float> training, ITrainingDataSource<float> validation, ModuleTrainingEnumerator.Report report);
    public string FormatOutput(string inputStr, Tensor<float> input, Tensor<float> output);
}

public abstract class Example : IExample
{
    public virtual string Name => GetType().Name;
    public virtual void Configure(string json) { }
    protected virtual string ExamplePath => Path.Combine(Environment.CurrentDirectory, "Examples", Name);
    protected virtual string ProcessedDataPath => Path.Combine(ExamplePath, "tensors");
    protected virtual string RawDataPath => Path.Combine(ExamplePath, "raws");

    protected const string DefaultWeightsFilename = "Network.safetensors";

    public virtual void Clean()
    {
        if (Directory.Exists(ProcessedDataPath))
        {
            foreach (var file in Directory.EnumerateFiles(ProcessedDataPath))
            {
                File.Delete(file);
            }
        }
    }

    public abstract INetworkModule GetArchitecture();
    public virtual Safetensors LoadWeights()
    {
        return Safetensors.ReadFromFile(Path.Combine(ExamplePath, DefaultWeightsFilename));
    }
    public virtual void SaveWeights(Safetensors tensors)
    {
        tensors.WriteToFile(Path.Combine(ExamplePath, DefaultWeightsFilename));
    }

    public abstract Tensor<float> ParseUserInput(string input);

    public virtual void ProcessRawData() {}
    public abstract void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation);
    public virtual ITrainingDataSampler<float> GetTrainingSampler(ITrainingDataSource<float> training) => training.CreateRandomSampler(allowDuplicates: false);
    public virtual ITrainingDataSampler<float> GetValidationSampler(ITrainingDataSource<float> validation) => validation.CreateSequentialSampler();
    public abstract void ConfigureTrainer(ModuleTrainer trainer);
    public virtual IEnumerable<IReport> GenerateTrainingReports(INetworkModule network, ITrainingDataSource<float> training, ITrainingDataSource<float> validation, ModuleTrainingEnumerator.Report report) => Enumerable.Empty<IReport>();
    public abstract string FormatOutput(string inputStr, Tensor<float> input, Tensor<float> output);
}