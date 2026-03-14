using System.Collections;

namespace DotML.Network.Training;

/// <summary>
/// Flexible definition for amount of each kind of member for a given population
/// </summary>
public struct PopulationProportion
{
    // TODO sensible defaults
    /// <summary>
    /// Number of elite performers
    /// </summary>
    public uint Elite  = 25;         // 25%   
    /// <summary>
    /// Number of crossovers
    /// </summary>
    public uint Crossover = 50;     // 50%
    /// <summary>
    /// Number of mutations of the elite performers
    /// </summary>
    public uint EliteMutations = 20; // 20%
    /// <summary>
    /// Number of randomly generated
    /// </summary>
    public uint Random = 5;         // 5%    

    public PopulationProportion()
    {
        this.Elite = 25;
        this.Crossover = 50;
        this.EliteMutations = 20;
        this.Random = 5;
    }

    public PopulationProportion(uint elite, uint crossover, uint eliteMutations = 0, uint random = 0)
    {
        this.Elite = elite;
        this.Crossover = crossover;
        this.EliteMutations = eliteMutations;
        this.Random = random;
    }

    private float FlexSum => Math.Max(1, Elite + Crossover + Random + EliteMutations); // Guaranteed to be >= 0 due to uints for the above ^^ 

    /// <summary>
    /// Percentage of the population that is elite [0-1]
    /// </summary>
    public float PercentElite => (float)Elite / FlexSum;
    /// <summary>
    /// Percentage of population that is crossover [0-1]
    /// </summary>
    public float PercentCrossover => (float)Crossover / FlexSum;
    /// <summary>
    /// Percentage of population that is random [0-1]
    /// </summary>
    public float PercentRandom => (float)Random / FlexSum;
    /// <summary>
    /// Percent of population that is a mutation of the elite [0-1]
    /// </summary>
    public float PercentMutation => (float)EliteMutations / FlexSum;
}