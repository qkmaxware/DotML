namespace DotML.Network.Training;

public interface IValidationReport {
    public int SampleCount {get;}

    public Metric<float> Loss {get;}
}