namespace DotML.NetDot.Dot;

public class DotVertex
{
    public string? Id { get; init; }
    public Dictionary<string, string> Attributes { get; init; } = new();

    public DotVertex() { }

    public DotVertex(string id)
    {
        this.Id = id;
    }
}

public class DotEdge
{
    public Dictionary<string, string> Attributes { get; init; } = new();
}

public enum DotGraphMode
{
    Undirected, Directed
}

public class DotGraph : AdjacencyListGraph<DotVertex, DotEdge>
{
    public string? Id { get; set; }
    public DotGraphMode Mode { get; set; }
    public Dictionary<string, string> Attributes { get; init; } = new();

    public DotVertex GetOrAdd(string Id, Func<string, DotVertex> builder)
    {
        var existing = this.FindVertex(node => Id == node.Id);
        if (existing is not null)
            return existing;

        var node = builder(Id);
        this.Add(node);
        return node;
    }

    public Edge GetOrConnect(DotVertex from, DotVertex to, Action<Edge> builder)
    {
        var existing = this.GetEdge(from, to);
        if (existing is not null)
            return existing;

        var edge = this.Connect(from, to, new DotEdge());
        if (edge is null)
            throw new ArgumentException("Cannot create an edge between the given vertices, they may not exist in the graph");
        builder(edge);
        return edge;
    }
}