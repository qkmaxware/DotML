namespace DotML.Network.Training;

/// <summary>
/// A network training schedule
/// </summary>
/// <typeparam name="T">Network type</typeparam>
public interface ITrainingSchedule<T> where T:INeuralNetwork {
    /// <summary>
    /// Begin training the following networks
    /// </summary>
    /// <param name="networks">List of networks to train</param>
    public void BeginSchedule(T[] networks);
}

/// <summary>
/// Training schedule where multiple networks are trained in sequence
/// </summary>
/// <typeparam name="T">Network type</typeparam>
public class SequentialTraining<T> : ITrainingSchedule<T> where T:INeuralNetwork {

    /// <summary>
    /// The trainer used in this schedule
    /// </summary>
    public ITrainer<T> Trainer {get; set;}
    /// <summary>
    /// The loss function used to verify the accuracy of the trained models
    /// </summary>
    public LossFunction LossFunction {get; set;} = LossFunctions.MeanSquaredError;
    /// <summary>
    /// The data used to train the networks, shared between all networks being trained
    /// </summary>
    public TrainingSet TrainingRegimine {get; set;}
    /// <summary>
    /// The data used to test the accuracy of the trained networks. 
    /// </summary>
    public TrainingSet? TestingCriteria {get; set;}

    public SequentialTraining(ITrainer<T> trainer, TrainingSet regimine, TrainingSet? validation = null) {
        this.Trainer = trainer;
        this.TrainingRegimine = regimine;
        this.TestingCriteria = validation;
    }

    /// <summary>
    /// Begin training the following networks
    /// </summary>
    /// <param name="networks">List of networks to train</param>
    public void BeginSchedule(T[]  networks) {
        foreach (var network in networks) {
            Trainer.Train(network, TrainingRegimine.SampleRandomly(), (TestingCriteria ?? TrainingRegimine).SampleSequentially());
        }
    }

    /// <summary>
    /// Begin training the following networks, and return the "best" of the trained networks
    /// </summary>
    /// <param name="networks">List of networks to train</param>
    /// <returns>Best trained network</returns>
    public T BeginScheduleReturningBest(T[] networks) {
        BeginSchedule(networks);

        var data = TestingCriteria ?? TrainingRegimine;
        var datasize = data.Size;
        var loss = 0.0;
        T best = networks[0];
        bool first = true;
        foreach (var network in networks) {
            var set = data.SampleSequentially();
            var netloss = 0.0;
            while (set.MoveNext()) {
                var predicted = network.PredictSync(set.Current.Input);
                netloss += LossFunction(predicted, set.Current.Output);
            }
            netloss /= datasize;

            if (first || netloss < loss) {
                loss = netloss;
                best = network;
            }
            first = false;
        }

        return best;
    }
}

/// <summary>
/// Training schedule where multiple networks are trained in parallel using Parallel.For
/// </summary>
/// <typeparam name="T">Network type</typeparam>
public class ParallelTraining<T> : ITrainingSchedule<T> where T:INeuralNetwork {
    /// <summary>
    /// The trainer used in this schedule
    /// </summary>
    public ITrainer<T> Trainer {get; set;}
    /// <summary>
    /// The loss function used to verify the accuracy of the trained models
    /// </summary>
    public LossFunction LossFunction {get; set;} = LossFunctions.MeanSquaredError;
    /// <summary>
    /// The data used to train the networks, shared between all networks being trained
    /// </summary>
    public TrainingSet TrainingRegimine {get; set;}
    /// <summary>
    /// The data used to test the accuracy of the trained networks. 
    /// </summary>
    public TrainingSet? TestingCriteria {get; set;}

    public ParallelTraining(ITrainer<T> trainer, TrainingSet regimine, TrainingSet? validation = null) {
        this.Trainer = trainer;
        this.TrainingRegimine = regimine;
        this.TestingCriteria = validation;
    }

    /// <summary>
    /// Begin training the following networks
    /// </summary>
    /// <param name="networks">List of networks to train</param>
    public void BeginSchedule(T[] networks) {
        Parallel.For(0, networks.Length, (i) => {
            Trainer.Train(networks[i], TrainingRegimine.SampleRandomly(), (TestingCriteria ?? TrainingRegimine).SampleSequentially());
        });
    }

    /// <summary>
    /// Begin training the following networks, and return the "best" of the trained networks
    /// </summary>
    /// <param name="networks">List of networks to train</param>
    /// <returns>Best trained network</returns>
    public T BeginScheduleReturningBest(T[] networks) {
        BeginSchedule(networks);

        var data = TestingCriteria ?? TrainingRegimine;
        var datasize = data.Size;
        var loss = 0.0;
        T best = networks[0];
        bool first = true;
        foreach (var network in networks) {
            var set = data.SampleSequentially();
            var netloss = 0.0;
            while (set.MoveNext()) {
                var predicted = network.PredictSync(set.Current.Input);
                netloss += LossFunction(predicted, set.Current.Output);
            }
            netloss /= datasize;

            if (first || netloss < loss) {
                loss = netloss;
                best = network;
            }
            first = false;
        }

        return best;
    }
}