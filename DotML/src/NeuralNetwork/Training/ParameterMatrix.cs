using System.Collections;

namespace DotML.Network.Training;

/*
Desired usage

// Create a parameter matrix
var parameters = new ParameterMatrix(
    "epoch": Enumerable.Range(200, 1000).ToArray(),
    "learning_rate": [ 0.001, 0.01, 0.1 ],
    "hidden": Enumerable.Range(27, 72).ToArray()
);

// Enumerate over cross-join of all parameters
// Create a matrix and trainer from the cross-joined parameters
var networks = parameters.Select(set => (
    return (
        Network: new ClassicalNeuralNetwork(72, set.Get<int>("hidden"), 26),
        Trainer: new EnumerableBackpropagationTrainer<ClassicalNeuralNetwork>() {
            Epochs: set.Get<int>("epoch"),
            LearningRate: set.Get<double>("learning_rate")
        };
    )
)).ToArray();

// Train them all at the same time with all possible combinations :)
TrainingSet training;
TrainingSet validation;
Parallel.For(0, networks.Length, (index) => {
    networks[i].Trainer.Train(networks[i].Network, training, validation);
});

// Determine the "best" training/network parameters from the best network
*/

public class ParameterSet : Dictionary<string, object> {
    public ParameterSet() : base() {}
    public ParameterSet(int capacity) : base(capacity) { }
    public ParameterSet(ParameterSet other) : base(other) { }

    public T Get<T>(string key) {
        var value = this[key];
        if (value.GetType().IsAssignableTo(typeof(T))) {
            return (T)value;
        } else {
            throw new InvalidCastException(key);
        }
    }

    public T Get<T>(string key, T @default) {
        var value = this[key];
        if (value.GetType().IsAssignableTo(typeof(T))) {
            return (T)value;
        } else {
            return @default;
        }
    }
}

public class ParameterMatrix : IEnumerable<ParameterSet> {
    private List<KeyValuePair<string, object[]>> matrix_values;
    public ParameterMatrix(Dictionary<string, object[]> matrix_values) {
        this.matrix_values = matrix_values.ToList();
    }  

    public ParameterMatrix(params KeyValuePair<string, object[]>[] matrix_values) {
        this.matrix_values = new List<KeyValuePair<string, object[]>>(matrix_values);
    } 

    public ParameterMatrix(params (string, object[])[] args) {
        matrix_values = new List<KeyValuePair<string, object[]>>(args.Length);

        for (int i = 0; i < args.Length; i ++) {
            var arg = args[i];
            matrix_values.Add(new KeyValuePair<string, object[]>(arg.Item1, arg.Item2));
        }
    } 

    private static IEnumerable<ParameterSet> CrossJoin(ParameterSet current_join, List<KeyValuePair<string, object[]>> arrays, int depth) {
        if (depth >= arrays.Count) {
            yield return new ParameterSet(current_join);
            yield break;
        }

        var current = arrays[depth];
        var key = current.Key;
        var values = current.Value;
        foreach (var item in values) {
            current_join[key] = item;

            foreach (var combination in CrossJoin(current_join, arrays, depth + 1)) {
                yield return combination;
            }
        }

        current_join.Remove(key);
    }

    public IEnumerator<ParameterSet> GetEnumerator() => CrossJoin(new ParameterSet(), matrix_values, 0).GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();
}
