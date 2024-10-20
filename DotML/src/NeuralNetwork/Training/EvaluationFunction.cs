namespace DotML.Network.Training;

/// <summary>
/// Delegate representing a function that evaluates the accuracy of a network
/// </summary>
/// <param name="network">The network to evaluate</param>
/// <param name="validation">The data to evaluate the network against</param>
/// <returns>loss evaluation</returns>
public delegate double NetworkEvaluationFunction(INeuralNetwork network, TrainingSet validation);

/// <summary>
/// Container with some common network evaluation functions
/// </summary>
public static class NetworkEvaluationFunctions {
    /// <summary>
    /// Evaluate the maximum mean squared error of a network across an entire training set
    /// </summary>
    /// <param name="network">The network to evaluate</param>
    /// <param name="validation">The data to evaluate the network against</param>
    /// <returns>maximum loss experienced by the network</returns>
    public static double MaxMeanSquaredError(INeuralNetwork network, TrainingSet validation) {
        var data = validation;
        var maxloss = 0.0;

        var set = data.SampleSequentially();
        while (set.MoveNext()) {
            var predicted = network.PredictSync(set.Current.Input);
            maxloss = Math.Max(maxloss, LossFunctions.MeanSquaredError(predicted, set.Current.Output));
        }

        return maxloss;
    }

    /// <summary>
    /// Evaluate the maximum mean squared error of a network across an entire training set
    /// </summary>
    /// <param name="network">The network to evaluate</param>
    /// <param name="validation">The data to evaluate the network against</param>
    /// <returns>maximum loss experienced by the network</returns>
    public static double MaxMeanSquaredError(INeuralNetwork network, IEnumerator<TrainingPair> validation) {
        var maxloss = 0.0;

        var set = validation;
        set.Reset();
        while (set.MoveNext()) {
            var predicted = network.PredictSync(set.Current.Input);
            maxloss = Math.Max(maxloss, LossFunctions.MeanSquaredError(predicted, set.Current.Output));
        }

        return maxloss;
    }

    /// <summary>
    /// Evaluate the average mean squared error of a network across an entire training set
    /// </summary>
    /// <param name="network">The network to evaluate</param>
    /// <param name="validation">The data to evaluate the network against</param>
    /// <returns>average loss experienced by the network</returns>
    public static double AvgMeanSquaredError(INeuralNetwork network, TrainingSet validation) {
        var data = validation;
        var datasize = data.Size;
        var netloss = 0.0;

        var set = data.SampleSequentially();
        while (set.MoveNext()) {
            var predicted = network.PredictSync(set.Current.Input);
            netloss += LossFunctions.MeanSquaredError(predicted, set.Current.Output);
        }
        netloss /= datasize;

        return netloss;
    }

    /// <summary>
    /// Evaluate the average mean squared error of a network across an entire training set
    /// </summary>
    /// <param name="network">The network to evaluate</param>
    /// <param name="validation">The data to evaluate the network against</param>
    /// <returns>average loss experienced by the network</returns>
    public static double AvgMeanSquaredError(INeuralNetwork network, IEnumerator<TrainingPair> validation) {
        var datasize = 0;
        var netloss = 0.0;

        var set = validation;
        set.Reset();
        while (set.MoveNext()) {
            var predicted = network.PredictSync(set.Current.Input);
            netloss += LossFunctions.MeanSquaredError(predicted, set.Current.Output);
            datasize++;
        }
        netloss /= datasize;

        return netloss;
    }
}