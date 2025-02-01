using System.Collections;

namespace DotML.Network.Training;

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
