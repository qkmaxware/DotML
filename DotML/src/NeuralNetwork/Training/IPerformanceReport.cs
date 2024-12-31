using System.Collections.Concurrent;
using System.Diagnostics;

namespace DotML.Network.Training;

public interface IBenchmarkClock : IDisposable { }

public interface IPerformanceReport {
    public IBenchmarkClock? Begin(string uid);
}

public class DefaultBenchmarkClock : IBenchmarkClock {
    private string key;
    private ConcurrentDictionary<string, Benchmark> storage;
    private Stopwatch stopwatch;

    public DefaultBenchmarkClock(string key, ConcurrentDictionary<string, Benchmark> storage) {
        stopwatch = Stopwatch.StartNew();
        this.key = key;
        this.storage = storage;
    }

    public void Dispose() {
        stopwatch.Stop();
        var time = stopwatch.Elapsed;
        storage.AddOrUpdate(
            key, 
            new Benchmark { 
                Name = key,
                Max = time, 
                Min = time, 
                Sum = time, 
                Count = 1 
            }, 
            (k, old) => {
                if (time < old.Min) {
                    old.Min = time;
                }
                if (time > old.Max) {
                    old.Max = time;
                }
                old.Sum += time;
                old.Count += 1;
                return old;
            }
        );
    }
}

public class Benchmark {
    public string? Name {get; set;}
    public TimeSpan Min {get; set;}
    public TimeSpan Max {get; set;}
    public TimeSpan Sum {get; set;}
    public TimeSpan Average => Sum / Count;
    public int Count {get; set;}
}


public class DefaultPerformanceReport : IPerformanceReport {

    ConcurrentDictionary<string, Benchmark> benchmarks = new ConcurrentDictionary<string, Benchmark>(); 

    public IBenchmarkClock? Begin(string uid) {
        return new DefaultBenchmarkClock(uid, benchmarks);
    }

    public IEnumerable<Benchmark> Benchmarks => benchmarks.Values;
}