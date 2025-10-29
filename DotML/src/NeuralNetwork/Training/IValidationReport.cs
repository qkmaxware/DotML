namespace DotML.Network.Training;

public interface IValidationReport {
    public int SampleCount {get;}
    public int TestsPassedCount {get;}
    public int TestsFailedCount {get;}

    public Metric<float> Loss {get;}
}