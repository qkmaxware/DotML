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

    public int TruePositives {get; private set;}
    public int TrueNegatives {get; private set;}
    public int FalsePositives {get; private set;}
    public int FalseNegatives {get; private set;}

    /// <summary>
    /// Accuracy is the proportion of correctly predicted instances (both true positives and true negatives) to the total instances in the dataset.
    /// </summary>
    public double Accuracy => ((double)(TruePositives + TrueNegatives)) / ((double)test_all_count);
    /// <summary>
    /// Precision measures the proportion of correctly predicted positive instances (true positives) out of all the instances that were predicted as positive.
    /// </summary>
    public double Precision => ((double)TruePositives) / ((double)(TruePositives + FalsePositives));
    /// <summary>
    /// Recall measures the proportion of correctly predicted positive instances (true positives) out of all the actual positive instances.
    /// </summary>
    public double Recall => ((double)TruePositives) / ((double)(TruePositives + FalseNegatives));
    /// <summary>
    /// The F1 score is the harmonic mean of precision and recall. It provides a single score that balances both the concerns of precision and recall, especially when you need a balance between the two. It ranges from 0 to 1, where 1 is the best value.
    /// </summary>
    public double F1Score => 2.0 * (Precision * Recall) / (Precision + Recall);

    public double TrueProbabilityThreshold {get; set;} = 0.5;

    public virtual void Reset() {
        this.test_all_count = 0;
        this.test_passed_count = 0;
        this.test_failed_count = 0;

        this.sum_loss = 0;
        this.min_loss = 0;
        this.max_loss = 0;

        this.TruePositives = 0;
        this.FalsePositives = 0;
        this.TrueNegatives = 0;
        this.FalseNegatives = 0;
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

        // Compute true positives, false positives, true negatives, and false negatives
        // This assumes the outputs are probability distributions, which tbf they usually are
        var class_belonging_to = predicted.IndexOfMaxValue();  // Class label (index)
        var predicted_positive = Math.Clamp(predicted[class_belonging_to], 0.0, 1.0) > TrueProbabilityThreshold;

        var class_suppose_to = expected.IndexOfMaxValue(); // Class label (index)
        var actual_positive = Math.Clamp(expected[class_suppose_to], 0.0, 1.0) > TrueProbabilityThreshold;
  
        if (class_belonging_to == class_suppose_to) {
            // Predicted and actual class are the same, so it's either TP or TN
            if (predicted_positive) {
                // True Positive (correctly predicted as positive)
                this.TruePositives++;
            } else {
                // True Negative (correctly predicted as negative)
                this.TrueNegatives++;
            }
        } else {
            // Predicted class is different from actual class, so it's either FP or FN
            if (predicted_positive) {
                // False Positive (predicted positive, but actual is negative)
                this.FalsePositives++;
            } else {
                // False Negative (predicted negative, but actual is positive)
                this.FalseNegatives++;
            }
        }
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