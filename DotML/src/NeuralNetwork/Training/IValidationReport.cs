namespace DotML.Network.Training;

public interface IValidationReport {

    public int TestCount {get;}
    public int TestsPassedCount {get;}
    public int TestsFailedCount {get;}

    public double MaxLoss {get;}
    public double MinLoss {get;}
    public double AverageLoss {get;}
    public void Reset();
    public void Append(Vec<double> input, Vec<double> expected, Vec<double> predicted, bool testPassed, double loss);
}

public struct TestBreakdown {
    public int Index {get; set;}
    public double Loss {get; set;}
    public bool Passed {get; set;}
    public Vec<double> Input {get; set;}
    public Vec<double> Expected {get; set;}
    public Vec<double> Predicted {get; set;}
}

public interface IValidationReportWithTestBreakdown : IValidationReport {
    public IEnumerable<TestBreakdown> TestBreakdown {get;}
}

public class DefaultValidationReport : IValidationReport {

    private int test_all_count;
    private int test_passed_count;
    private int test_failed_count;

    private double sum_loss;
    private double min_loss;
    private double max_loss;

    public int TestCount => test_all_count;
    public int TestsPassedCount => test_passed_count;
    public int TestsFailedCount => test_failed_count;

    public double MaxLoss => max_loss;
    public double MinLoss => min_loss;
    public double AverageLoss => sum_loss / test_all_count;

    public virtual void Reset() {
        this.test_all_count = 0;
        this.test_passed_count = 0;
        this.test_failed_count = 0;

        this.sum_loss = 0;
        this.min_loss = 0;
        this.max_loss = 0;
    }

    public virtual void Append(Vec<double> input, Vec<double> expected, Vec<double> predicted, bool testPassed, double loss) {
        this.sum_loss += loss;
        if (test_all_count == 0 || loss > max_loss)
            max_loss = loss;
        if (test_all_count == 0 || loss < min_loss)
            min_loss = loss;

        test_all_count++;
        if (testPassed)
            test_passed_count++;
        else 
            test_failed_count++;
    }

}

public class DefaultValidationReportWithBreakdown : DefaultValidationReport, IValidationReportWithTestBreakdown {
    
    private List<TestBreakdown> breakdowns = new List<TestBreakdown>();
    public IEnumerable<TestBreakdown> TestBreakdown => breakdowns;
    
    public override void Reset() {
        base.Reset();

        breakdowns.Clear();
    }

    public override void Append(Vec<double> input, Vec<double> expected, Vec<double> predicted, bool testPassed, double loss) {
        base.Append(input, expected, predicted, testPassed, loss);

        breakdowns.Add(new TestBreakdown {
            Index = breakdowns.Count,
            Loss = loss,
            Passed = testPassed,
            Input = input,
            Expected = expected,
            Predicted = predicted
        });
    }
}