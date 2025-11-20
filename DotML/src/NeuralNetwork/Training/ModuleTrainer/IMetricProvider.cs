namespace DotML.Network.Training;

/// <summary>
/// A provider which allows access to computed metrics
/// </summary>
public interface IMetricsProvider
{
    /// <summary>
    /// Reset all metrics
    /// </summary>
    public void Reset();
    
    /// <summary>
    /// Add a sample to the cached metrics
    /// </summary>
    /// <param name="loss">loss</param>
    /// <param name="predicted">predicted values</param>
    /// <param name="truth">ground truth values</param>
    public void AddSample(float loss, ReadOnlySpan<float> predicted, ReadOnlySpan<float> truth);
}

/// <summary>
/// Metrics provider for Accuracy, Precision, Recall, F1, based on a Confusion Matrix
/// </summary>
public class AccuracyMetricsProvider : IMetricsProvider
{
    /// <summary>
    /// Number of samples checked
    /// </summary>
    private int sampleCount {get; set;}
    /// <summary>
    /// Number of samples with the correct labels
    /// </summary>
    public int TestsPassedCount { get; set; }
    /// <summary>
    /// Number of samples with the incorrect labels
    /// </summary>
    public int TestsFailedCount => sampleCount - TestsPassedCount;
    /// <summary>
    /// Number of class labels
    /// </summary>
    public int NumberOfClasses;
    /// <summary>
    /// Accuracy of the training iteration based on the number of correct samples
    /// </summary>
    public float Accuracy => sampleCount > 0 ? (float)TestsPassedCount / sampleCount : 0f;
    /// <summary>
    /// Precision of the training iteration based on the number of true positives and false positives
    /// </summary>
    public float Precision
    {
        get
        {
            int classes = confusionMatrix.GetLength(0);
            float totalPrecision = 0f;
            int validClasses = 0;

            for (int k = 0; k < classes; k++)
            {
                int tp = confusionMatrix[k, k];
                int fp = 0;

                for (int i = 0; i < classes; i++)
                {
                    if (i == k)
                        continue;

                    fp += confusionMatrix[i, k];
                }

                int denominator = tp + fp;
                if (denominator > 0)
                {
                    totalPrecision += (float)tp / denominator;
                    validClasses++;
                }
            }

            return validClasses > 0 ? totalPrecision / validClasses : 0f;
        }
    }
    /// <summary>
    /// Recall of the training iteration based on the number of true positives and false negatives 
    /// </summary>
    public float Recall
    {
        get
        {
            int classes = confusionMatrix.GetLength(0);
            float totalRecall = 0f;
            int validClasses = 0;

            for (int k = 0; k < classes; k++)
            {
                int tp = confusionMatrix[k, k];
                int fn = 0;

                for (int j = 0; j < classes; j++)
                {
                    if (j == k)
                        continue;

                    fn += confusionMatrix[k, j];
                }

                int denominator = tp + fn;
                if (denominator > 0)
                {
                    totalRecall += (float)tp / denominator;
                    validClasses++;
                }
            }

            return validClasses > 0 ? totalRecall / validClasses : 0f;
        }
    }
    /// <summary>
    /// F1 score of the training iteration based on precision and recall
    /// </summary>
    public float F1
    {
        get
        {
            var precision = this.Precision;
            var recall = this.Recall;
            if (precision + recall <= 0)
                return 0.0f;

            return 2f * precision * recall / (precision + recall);
        }
    }

    private int[,] confusionMatrix = new int[0, 0];

    public int[,] GetConfusionMatrix() 
    {
        return this.confusionMatrix;
    }

    public void Reset()
    {
        sampleCount = 0;

        TestsPassedCount = 0;
        NumberOfClasses = 0;

        for (var i = 0; i < confusionMatrix.GetLength(0); i++)
            for (var j = 0; j < confusionMatrix.GetLength(0); j++)
                confusionMatrix[i, j] = 0;
    }

    public void AddSample(float loss, ReadOnlySpan<float> predicted, ReadOnlySpan<float> truth)
    {
        this.sampleCount++;

        int predictedClass = ArgMax(predicted);
        int trueClass = ArgMax(truth);
        if (predictedClass == trueClass)
            TestsPassedCount++;

        var numClasses = truth.Length; // Assumes one-hot encoding and no multi-classes
        NumberOfClasses = numClasses;

        // Populate the confusion matrix (make a new matrix if required)
        if (confusionMatrix is null || confusionMatrix.GetLength(0) < numClasses)
            confusionMatrix = new int[numClasses, numClasses];

        confusionMatrix[trueClass, predictedClass]++;
    }

    private static int ArgMax(ReadOnlySpan<float> vector)
    {
        int maxIndex = 0;
        float maxVal = vector[0];
        for (int i = 1; i < vector.Length; i++)
        {
            if (vector[i] > maxVal)
            {
                maxVal = vector[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}

/// <summary>
/// Metrics provider for SNR and PSNR both globally across all samples, and per-sample
/// </summary>
public class SignalToNoiseProvider : IMetricsProvider
{
    private double sumSignalSquared;     // Sum truth[i]^2 over all samples
    private double sumNoiseSquared;      // Sum (truth[i]-pred[i])^2 over all samples
    private double maxSignalValue;       // Max truth value seen in total (for PSNR)
    private int totalCount;

    public double GlobalSNR
    {
        get
        {
            if (sumNoiseSquared == 0)
                return float.PositiveInfinity;

            return 10.0 * Math.Log10(sumSignalSquared / sumNoiseSquared);
        }
    }

    public double GlobalPSNR
    {
        get
        {
            if (totalCount == 0) 
                return double.NaN;

            double mse = sumNoiseSquared / totalCount;

            if (mse == 0)
                return double.PositiveInfinity;

            return 10.0 * Math.Log10((maxSignalValue * maxSignalValue) / mse);
        }
    }

    private double? maxSampleSignal = 0;
    public Metric<double> PerSampleSNR {get; private set;} = new Metric<double>();

    public Metric<double> PerSamplePSNR {get; private set;} = new Metric<double>();

    public SignalToNoiseProvider(double? maxSampleSignal)
    {
        this.maxSampleSignal = maxSampleSignal.HasValue ? Math.Abs(maxSampleSignal.Value) : null;
    }

    public void Reset()
    {
        sumSignalSquared = 0;
        sumNoiseSquared = 0;
        maxSignalValue = 0;
        totalCount = 0;

        this.PerSampleSNR.Reset();
        this.PerSamplePSNR.Reset();
    }

    public void AddSample(float loss, ReadOnlySpan<float> predicted, ReadOnlySpan<float> truth)
    {
        double signal = 0;
        double noise = 0;
        double localMax = maxSampleSignal.HasValue ? maxSampleSignal.Value : 0;
        int localCount = 0;

        var len = Math.Min(predicted.Length, truth.Length);
        if (len == 0)
            return;

        for (int i = 0; i < len; i++)
        {
            double t = truth[i];
            double p = predicted[i];
            double diff = t - p;

            double tt = t * t;
            sumSignalSquared += tt;
            signal += tt;
            double diffdiff = diff * diff;
            noise += diffdiff;
            sumNoiseSquared  += diffdiff;

            if (t > maxSignalValue)
                maxSignalValue = t;
            if (!maxSampleSignal.HasValue && t > localMax)
                localMax = t;

            totalCount++;
            localCount++;
        }

        if (noise == 0)
        {
            PerSampleSNR.Add(double.PositiveInfinity);
            PerSamplePSNR.Add(double.PositiveInfinity);
        }
        else
        {
            double snr  = 10.0 * Math.Log10(signal / noise);
            double mse  = noise / localCount;
            double psnr = 10.0 * Math.Log10((localMax * localMax) / mse);

            PerSampleSNR.Add(snr);
            PerSamplePSNR.Add(psnr);
        }
    }
}

/// <summary>
/// Computes SSIM for single-channel (Y) images.
/// Compatible with flattened 1HW row-major spans.
/// </summary>
public class StructuralSimilarityIndexProvider : IMetricsProvider
{
    /// <summary>
    /// Per-sample SSIM tracked for statistics (mean, min, max, etc.)
    /// </summary>
    public Metric<double> SSIM { get; private set; } = new Metric<double>();

    private int totalCount;

    private double maxValue;
    private double C1;
    private double C2;

    public StructuralSimilarityIndexProvider(double maxValue, double k1=0.01, double k2=0.03)
    {
        this.maxValue = maxValue;
        this.C1 = (k1 * maxValue) * (k1 * maxValue);
        this.C2 = (k2 * maxValue) * (k2 * maxValue);
    }

    /// <summary>
    /// Reset all metrics
    /// </summary>
    public void Reset()
    {
        SSIM.Reset();
        totalCount = 0;
    }

    /// <summary>
    /// Add a sample to the cached metrics
    /// </summary>
    /// <param name="loss">loss (can be ignored here)</param>
    /// <param name="predicted">predicted Y channel span</param>
    /// <param name="truth">ground truth Y channel span</param>
    public void AddSample(float loss, ReadOnlySpan<float> predicted, ReadOnlySpan<float> truth)
    {
        // Ensure spans are the same length
        var len = Math.Min(predicted.Length, truth.Length);
        if (len == 0)
            return;

        var ssim = ComputeSsim(predicted.Slice(0, len), truth.Slice(0, len));

        // Track per-sample SSIM
        SSIM.Add(ssim);

        // Update global SSIM as running mean
        //GlobalSSIM = (GlobalSSIM * totalCount + ssim) / (totalCount + 1);
        totalCount++;
    }

    /// <summary>
    /// Compute SSIM for single-channel flattened spans
    /// </summary>
    private double ComputeSsim(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        int n = x.Length;
        if (n == 0)
            return double.NaN;

        // Compute means
        double meanX = 0, meanY = 0;
        for (int i = 0; i < n; i++)
        {
            meanX += x[i];
            meanY += y[i];
        }
        meanX /= n;
        meanY /= n;

        // Compute variances and covariance
        double varX = 0, varY = 0, covXY = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - meanX;
            double dy = y[i] - meanY;
            varX += dx * dx;
            varY += dy * dy;
            covXY += dx * dy;
        }

        var denom = Math.Max(n - 1, 1);
        varX /= denom;
        varY /= denom;
        covXY /= denom;

        double numerator = (2 * meanX * meanY + C1) * (2 * covXY + C2);
        const double eps = 1e-10;
        double denominator = (meanX*meanX + meanY*meanY + C1) * (varX + varY + C2 + eps);

        return numerator / denominator;
    }
}

