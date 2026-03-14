using System.Buffers;
using System.Collections;

namespace DotML.Network.Training;

/// <summary>
/// Factory for generating random genomes
/// </summary>
/// <returns>random genome</returns>
public delegate IGenome GenomeFactory();

/// <summary>
/// Fitness testing and scheduling for multiple genomes
/// </summary>
public interface IFitnessTestScheduler {
    /// <summary>
    /// Schedule and test each genome putting their computed fitness scores in the equivalent spot
    /// </summary>
    /// <param name="genomes">genomes to test</param>
    /// <param name="fitnesses">fitness scores to populate</param>
    public void Test(ReadOnlyMemory<IGenome> genomes, Memory<float> fitnesses);
}

/// <summary>
/// A fitness testing system when tests each genome individually, one after another
/// </summary>
public abstract class SequentialTestScheduler : IFitnessTestScheduler
{
    public void Test(ReadOnlyMemory<IGenome> genomes, Memory<float> fitnesses)
    {
        var genome_span = genomes.Span;
        var fitness_span = fitnesses.Span;

        for (var i = 0; i < genomes.Length; i++)
        {
            var genome = genome_span[i];
            var fitness = Test(genome);
            fitness_span[i] = fitness;
        }
    }

    /// <summary>
    /// Fitness test function to determine the fitness of a given genome
    /// </summary>
    /// <param name="genome">genome to test</param>
    /// <returns>fitness value</returns>
    public abstract float Test(IGenome genome);
}

/// <summary>
/// A fitness testing system where all genomes are tested in parallel
/// </summary>
public abstract class ParallelTestScheduler : IFitnessTestScheduler
{
    public void Test(ReadOnlyMemory<IGenome> genomes, Memory<float> fitnesses)
    {
        Parallel.For(0, genomes.Length, (i) =>
        {
            var genome = genomes.Span[i];
            var fitness = Test(genome);
            fitnesses.Span[i] = fitness;
        });
    }

    /// <summary>
    /// Fitness test function to determine the fitness of a given genome
    /// </summary>
    /// <param name="genome">genome to test</param>
    /// <returns>fitness value</returns>
    public abstract float Test(IGenome genome);
}

/// <summary>
/// A stop condition for evolution using a genetic trainer
/// </summary>
/// <param name="genome">best genome of generation</param>
/// <param name="fitness">the raw fitness value of the best genome</param>
/// <param name="fitnessRank">the fitness rank of the best genome</param>
/// <returns>true if evolution should stop with this genome</returns>
public delegate bool EvolutionStopCondition(IGenome genome, float fitness, int fitnessRank);

/// <summary>
/// Trainer to configure genetic algorithm training for a population
/// </summary>
public class GeneticTrainer
{
    /// <summary>
    /// Size of the population
    /// </summary>
    public int PopulationSize {get; set;}
    /// <summary>
    /// Proportion of the population belonging to each class
    /// </summary>
    public PopulationProportion Proportions {get; set;}
    /// <summary>
    /// Factory to produce random genomes
    /// </summary>
    public GenomeFactory? Factory {get; set;}
    /// <summary>
    /// Batch size used for fitness testing
    /// </summary>
    public int BatchSize { get; set; }
    /// <summary>
    /// Test to check the fitness of each genome
    /// </summary>
    public IFitnessTestScheduler? FitnessTest {get; set;}
    /// <summary>
    /// Max number of generations to evolve the population over
    /// </summary>
    public int? MaxGenerations {get; set;}

    /// <summary>
    /// Weighting for fitness in the combined score
    /// </summary>
    public float FitnessWeighting {get; set;} = 0.8f;
    /// <summary>
    /// Weighting for diversity in the combined score
    /// </summary>
    public float DiversityWeighting {get; set;} = 0.2f;

    /// <summary>
    /// Rate of mutation 
    /// </summary>
    public float MutationRate {get; set;} = 0.01f;
    /// <summary>
    /// The condition in which to stop the evolution process early. 
    /// </summary>
    public EvolutionStopCondition? StopCondition {get; set;}

    /// <summary>
    /// Get an enumerator over a population of genomes which can be looped over to progress each generation
    /// </summary>
    /// <returns>enumerator</returns>
    /// <exception cref="NullReferenceException"></exception>
    public GeneticTrainerEnumerator EnumerateTraining()
    {
        if (this.Factory is null)
            throw new NullReferenceException(nameof(this.Factory));
        if (this.FitnessTest is null)
            throw new NullReferenceException(nameof(this.FitnessTest));

        return new GeneticTrainerEnumerator(this.PopulationSize, this.Proportions, this.Factory, this.BatchSize, this.FitnessTest, StopCondition, this.MaxGenerations, MutationRate, FitnessWeighting, DiversityWeighting);
    }
    
    /// <summary>
    /// Run the evolution process until either MaxGenerations is reached or until the early stop condition is reached
    /// </summary>
    /// <returns></returns>
    public IGenome Train()
    {
        var looper = EnumerateTraining();
        looper.MoveEnd();
        return looper.Current.MostFit!;
    }

}

/// <summary>
/// Genetic algorithm for a population set
/// </summary>
public class GeneticTrainerEnumerator
: IEnumerator<GeneticTrainerEnumerator.PopulationReport>
{
    private struct GenomeFitness
    {
        public IGenome Genome;

        public float Fitness;
        public int FitnessRank;

        public float Diversity;
        public int DiversityRank;

        public float CombinedSelectionWeight;
        public float WeightedRank(float fitnessWeight, float diversityWeight) => FitnessRank * fitnessWeight + DiversityRank * diversityWeight;
    }

    private List<GenomeFitness> population;
    private List<GenomeFitness> nextPopulation;

    public class PopulationReport {
        public int Generation {get; set;}
        public IGenome? MostFit {get; set;}
        public Metric<float> Fitness {get; set;} = new Metric<float>();
    }

    /// <summary>
    /// Current population
    /// </summary>
    public PopulationReport Current {get; private set;} = new PopulationReport();

    object IEnumerator.Current => Current;

    public int Generation {get; private set;}

    public int PopulationSize {get; init;}
    public PopulationProportion Proportions {get; init;}
    public GenomeFactory Factory {get; init;}
    private int _batchSize = 1;
    /// <summary>
    /// Batch size used for fitness testing
    /// </summary>
    public int BatchSize {
        get => _batchSize;
        set => _batchSize = Math.Max(1, value);
    }
    public IFitnessTestScheduler FitnessTest {get; init;}
    public float MutationRate {get; init;} = 0.01f;
    public int? MaxGenerations {get; init;}

    public float FitnessWeighting {get; init;} = 0.8f;
    public float DiversityWeighting {get; init;} = 0.2f;

    public EvolutionStopCondition? StopCondition {get; init;}
    private bool bestFound = false;
    private Random rng;

    public GeneticTrainerEnumerator(int size, PopulationProportion proportion, GenomeFactory factory, int batch, IFitnessTestScheduler test, EvolutionStopCondition? stop, int? maxGenerations, float mutationRate, float fitnessWeight, float diversityWeight, Random? rng = null)
    {
        this.PopulationSize = Math.Max(1, size);
        this.Proportions = proportion;
        this.BatchSize = batch;
        this.Factory = factory;
        this.FitnessTest = test;
        this.MaxGenerations = maxGenerations.HasValue ? Math.Max(0, maxGenerations.Value) : null; // Ensure maxGenerations is positive
        this.rng = rng ?? Random.Shared;
        this.StopCondition = stop;
        
        this.MutationRate = mutationRate;
        this.FitnessWeighting = fitnessWeight;
        this.DiversityWeighting = diversityWeight;

        population = new List<GenomeFitness>(this.PopulationSize);
        nextPopulation = new List<GenomeFitness>(this.PopulationSize);

        Reset(); // Reset on creation
    }

    public void Dispose() { }

    private static int SortByFitness(GenomeFitness a, GenomeFitness b)
    {
        // Sort largest fitness
        return b.Fitness.CompareTo(a.Fitness);
    }

    private static int SortByDiversity(GenomeFitness a, GenomeFitness b)
    {
        // Sort largest diversity
        return b.Diversity.CompareTo(a.Diversity);
    }

    private int SortByCombinedRank(GenomeFitness a, GenomeFitness b)
    {
        // Low rank is good (it means highest fitness/diversity)
        return a.WeightedRank(FitnessWeighting, DiversityWeighting).CompareTo(b.WeightedRank(FitnessWeighting, DiversityWeighting));
    }

    private static float ComputeDiversity(IGenome a, IEnumerable<GenomeFitness> population)
    {
        float sum = 0;
        int count = 0;

        foreach (var b in population)
        {
            if (ReferenceEquals(a, b.Genome))
                continue;

            float distance = a.Dissimilarity(b.Genome); // Dissimilarity is a distance metric
            sum += distance;
            count++;
        }

        return (count > 0) ? (sum / count) : 0f;
    }

    private static GenomeFitness SelectByRankWeighted(Random rng, List<GenomeFitness> population, IGenome? excluding = null)
    {
        // First pass: compute total weight (1 / (1 + rank))
        float total = 0f;
        for (int i = 0; i < population.Count; i++)
        {
            total += population[i].CombinedSelectionWeight;
        }

        // Second pass: roulette selection
        float roll = (float)(rng.NextDouble() * total);
        float accum = 0f;

        for (int i = 0; i < population.Count; i++)
        {
            var row = population[i];
            accum += row.CombinedSelectionWeight;
            if (excluding is not null && ReferenceEquals(row.Genome, excluding))
                continue; // Can't select the excluded genome
            if (accum >= roll)
                return population[i];
        }

        return population[^1]; // fallback (rare)
    }

    public bool MoveNext()
    {
        // If we only go for a given number of generations, stop and don't even try to do another generation
        // If we already found the best (given a stop condition) stop and don't even try to do another generation
        if (bestFound || (MaxGenerations.HasValue && MaxGenerations.Value <= this.Generation))
            return false;

        // Compute next population counts
        int numElite = (int)Math.Floor(this.PopulationSize * Proportions.PercentElite);
        int numCrossover = (int)Math.Floor(this.PopulationSize * Proportions.PercentCrossover);
        if (numCrossover % 2 != 0)
        {
            numCrossover -= 1; // Num crossover is always a factor of 2
        }
        numCrossover = Math.Max(0, numCrossover);
        int numMutations = (int)Math.Floor(this.PopulationSize * Proportions.PercentMutation);
        int numRandom = Math.Max(0, this.PopulationSize - numCrossover - numMutations - numElite);
        this.nextPopulation.Clear();

        // Test fitness
        var invCount = 1.0f / (this.population.Count - 1);
        var genomes = ArrayPool<IGenome>.Shared.Rent(BatchSize);
        var fitnesses = ArrayPool<float>.Shared.Rent(BatchSize);
        try {
            for (var i = 0; i < this.population.Count; i += BatchSize) {
                // Copy genomes to buffer and reset the fitness scores
                var batchSize = Math.Min(BatchSize, this.population.Count - i);
                for (var j = 0; j < batchSize; j++)
                {
                    GenomeFitness row = this.population[i + j];
                    genomes[j] = row.Genome;
                    fitnesses[j] = 0.0f;
                }

                // Experiment (allows for parallel testing in the regimen)
                FitnessTest.Test(genomes.AsMemory(0, batchSize), fitnesses.AsMemory(0, batchSize));

                // Copy back fitnesses into population
                for (var j = 0; j < batchSize; j++)
                {
                    GenomeFitness row = this.population[i + j];
                    row.Fitness = fitnesses[j];
                    this.population[i + j] = row;
                }
            }
        } 
        finally
        {
            ArrayPool<IGenome>.Shared.Return(genomes);
            ArrayPool<float>.Shared.Return(fitnesses);
        }

        // Compute fitness rank for selection
        this.population.Sort(SortByFitness);
        for(int i = 0; i < this.population.Count; i++){
            var row = this.population[i];
            row.FitnessRank = i;
            this.population[i] = row;
        }

        // Stop condition
        var mostFit = this.population[0];
        if (StopCondition is not null && StopCondition(mostFit.Genome, mostFit.Fitness, mostFit.FitnessRank))
        {
            bestFound = true;
            return true; 
        }
        var report = this.Current;
        report.MostFit = mostFit.Genome;
        report.Fitness.Reset();
        foreach (var pop in this.population)
            report.Fitness.AddSample(pop.Fitness);
        report.Generation ++;
        this.Current = report;

        // Select elites
        for (var i = 0; i < numElite; i+=1)
        {
            IGenome elite = this.population[i].Genome; // Select the best performers (sorted by raw fitness)

            this.nextPopulation.Add(new GenomeFitness { Genome = elite });
        }

        // Compute diversity score
        for (int i = 0; i < this.population.Count; i++)
        {
            var row = this.population[i];
            row.Diversity = ComputeDiversity(row.Genome, this.population);
            this.population[i] = row;
        }

        // Rank by diversity
        this.population.Sort(SortByDiversity);
        for (int i = 0; i < this.population.Count; i++)
        {
            var row = this.population[i];
            row.DiversityRank = i;
            this.population[i] = row;
        }

        // Sort by *combined* rank for final selection order
        this.population.Sort(SortByCombinedRank);

        // Next population selection (crossover/mutation) by combined rank
        for (var i = 0; i < this.population.Count; i++)
        {
            // compute selection weight (1 / (1 + rank)) used by the SelectByRankWeighted function
            var item = population[i];
            item.CombinedSelectionWeight = 1f / (1 + item.WeightedRank(FitnessWeighting, DiversityWeighting));
            population[i] = item;
        }
        for (var i = 0; i < numCrossover; i+=2)
        {
            IGenome parentA = SelectByRankWeighted(rng, this.population).Genome;
            IGenome parentB = SelectByRankWeighted(rng, this.population, excluding: parentA).Genome; 

            var (childA, childB) = parentA.Crossover(parentB, this.MutationRate);
            this.nextPopulation.Add(new GenomeFitness{ Genome = childA });
            this.nextPopulation.Add(new GenomeFitness{ Genome = childB });
        }

        for (var i = 0; i < numMutations; i+=1)
        {
            IGenome elite = SelectByRankWeighted(rng, this.population).Genome; // Selected from elite

            var mutated = elite.Mutate(this.MutationRate);
            this.nextPopulation.Add(new GenomeFitness{ Genome = mutated });
        }

        // Fill the rest of the population with new random members (diversity add)
        for (var i = 0; i < numRandom; i+=1)
        {
            var genome = Factory(); // Randomly generated to fill population
            this.nextPopulation.Add(new GenomeFitness{ Genome = genome });
        }

        // Swap the population/next population buffers
        (this.population, this.nextPopulation) = (this.nextPopulation, this.population);

        this.Generation++;
        return true;
    }

    public void Reset() {
        this.Generation = 0;
        this.bestFound = false;

        this.Current = new PopulationReport
        {
            Generation = 0,

            MostFit = null,
            Fitness = new Metric<float>()
        };

        this.nextPopulation.Clear();
        this.population.Clear();
        for (var i = 0; i < this.PopulationSize; i++)
            this.population.Add(new GenomeFitness{ Genome = Factory() });
    }
}
