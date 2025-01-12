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

    // TODO more validation metrics
    /*
        Accuracy: Good for balanced datasets but may be misleading for imbalanced ones.
        Precision: Focuses on minimizing false positives.
        Recall: Focuses on minimizing false negatives.
        F1 Score: A balanced metric that considers both precision and recall.
    */
    /// <summary>
    /// Accuracy is the proportion of correctly predicted instances (both true positives and true negatives) to the total instances in the dataset.
    /// </summary>
    //public double Accuracy => true_positives + true_negatives / test_all_count;
    /// <summary>
    /// Precision measures the proportion of correctly predicted positive instances (true positives) out of all the instances that were predicted as positive.
    /// </summary>
    //public double Precision => true_positives / (true_positives + false_positives);
    /// <summary>
    /// Recall measures the proportion of correctly predicted positive instances (true positives) out of all the actual positive instances.
    /// </summary>
    //public double Recall => true_positives / (true_positives + false_negatives);
    /// <summary>
    /// The F1 score is the harmonic mean of precision and recall. It provides a single score that balances both the concerns of precision and recall, especially when you need a balance between the two. It ranges from 0 to 1, where 1 is the best value.
    /// </summary>
    //public double F1Score => 2 * (Precision * Recall) / (Precision + Recall);

    /*
2. Deriving TP, FP, TN, FN from Output
To compute these values, you need to compare the predicted output of your model with the true output. The approach depends on whether you're dealing with discrete class labels or probabilities:

a. For Discrete Class Labels (e.g., 0 or 1)
If your model outputs discrete class labels (e.g., 0 or 1), you can directly compare the predicted output with the true output.
For each data point, you check:
If the predicted class equals the true class (i.e., both are 1 or both are 0), it contributes to either TP or TN.
If the predicted class is 1 and the true class is 0, it contributes to FP.
If the predicted class is 0 and the true class is 1, it contributes to FN.
Example:

Predicted	Actual	TP	FP	TN	FN
1	        1	    1	0	0	0
0	        1	    0	0	1	1
1	        0	    0	1	0	0
0	        0	    0	0	1	0
b. For Probability Outputs
If your model outputs probabilities (e.g., the probability that a sample belongs to the positive class), you need to threshold the probabilities to decide the predicted class. A common threshold is 0.5, meaning:

If the predicted probability is greater than or equal to 0.5, classify the sample as positive (1).
If the predicted probability is less than 0.5, classify the sample as negative (0).
Once you threshold the probabilities, you can apply the same logic as in the discrete class labels case to calculate TP, FP, TN, and FN.
    */

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