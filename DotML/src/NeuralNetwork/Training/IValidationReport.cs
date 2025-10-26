namespace DotML.Network.Training;

public interface IValidationReport {
    public int SampleCount {get;}
    public int TestsPassedCount {get;}
    public int TestsFailedCount {get;}

    public float MaxLoss {get;}
    public float MinLoss {get;}
    public float AvgLoss {get;}
}