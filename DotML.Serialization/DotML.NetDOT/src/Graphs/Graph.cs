using System.Collections;
using System.Collections.ObjectModel;
using System.Data;
using System.Drawing;
using DotML.Network;
using Microsoft.VisualBasic;

namespace DotML.NetDot;

public class AdjacencyListGraph<TVertex, TEdgeData> where TVertex : notnull
{

    public record class Edge
    {
        /// <summary>
        /// Reference to the graph which owns this edge
        /// </summary>
        public AdjacencyListGraph<TVertex, TEdgeData> Graph { get; init; }
        /// <summary>
        /// The index of the vertex this edge starts at
        /// </summary>
        public int FromIndex { get; init; }
        /// <summary>
        /// The value of the vertex this edge starts at
        /// </summary>
        public TVertex From => Graph.GetVertex(FromIndex);
        /// <summary>
        /// The index of the vertex this edge ends at
        /// </summary>
        public int ToIndex { get; init; }
        /// <summary>
        /// The value of the vertex this edge ends at
        /// </summary>
        public TVertex To => Graph.GetVertex(ToIndex);
        /// <summary>
        /// The edge payload
        /// </summary>
        public TEdgeData? Data { get; set; }

        public Edge(AdjacencyListGraph<TVertex, TEdgeData> graph, int from, int to, TEdgeData? data = default(TEdgeData))
        {
            this.Graph = graph;
            this.FromIndex = from;
            this.ToIndex = to;
            this.Data = data;
        }
    }

    private List<TVertex> nodes;
    private List<List<Edge>> edges;
    private Dictionary<TVertex, int> indices;

    /// <summary>
    /// Enumerate over all vertices
    /// </summary>
    /// <returns>enumerable of vertices</returns>
    public IEnumerable<TVertex> EnumerateVertices() => nodes.AsReadOnly();

    /// <summary>
    /// Enumerate over all edges
    /// </summary>
    /// <returns>enumerable of edges</returns>
    public IEnumerable<Edge> EnumerateEdges() => edges.SelectMany(x => x);

    /// <summary>
    /// Create a new graph with default capacity
    /// </summary>
    public AdjacencyListGraph()
    {
        this.nodes = new List<TVertex>();
        this.edges = new List<List<Edge>>();
        this.indices = new Dictionary<TVertex, int>();
    }

    /// <summary>
    /// Create a new graph with the given capacity
    /// </summary>
    /// <param name="capacity">graph capacity</param>
    public AdjacencyListGraph(int capacity)
    {
        this.nodes = new List<TVertex>(capacity);
        this.edges = new List<List<Edge>>(capacity);
        this.indices = new Dictionary<TVertex, int>();
    }

    /// <summary>
    /// Number of vertices in the graph
    /// </summary>
    public int VertexCount => this.nodes.Count;

    /// <summary>
    /// Number of edges in the graph
    /// </summary>
    public int EdgeCount => this.edges.Sum(set => set.Count);

    /// <summary>
    /// Find a vertex with a given predicate condition
    /// </summary>
    /// <param name="predicate">predicate function</param>
    /// <returns>first node found or default</returns>
    public TVertex? FindVertex(Predicate<TVertex> predicate) => nodes.FirstOrDefault(node => predicate(node));

    /// <summary>
    /// Clear all vertices and edges from the graph
    /// </summary>
    public void Clear()
    {
        this.nodes.Clear();
        this.edges.Clear();
        this.indices.Clear();
    }

    /// <summary>
    /// Add a vertex to the graph
    /// </summary>
    /// <param name="data">vertex</param>
    public void Add(TVertex data)
    {
        if (this.indices.ContainsKey(data))
            throw new ArgumentException("Vertex already exists in the graph", nameof(data));

        var index = this.nodes.Count;
        this.nodes.Add(data);
        this.edges.Add(new List<Edge>());

        // Maintain indicies mapping
        this.indices[data] = index;
    }

    /// <summary>
    /// Remove a given vertex
    /// </summary>
    /// <param name="data">vertex</param>
    /// <returns>true if removal was successful</returns>
    public bool Remove(TVertex data)
    {
        if (!this.indices.TryGetValue(data, out int index))
            return false; // Not in graph

        return Remove(index);
    }

    /// <summary>
    /// Remove a given vertex
    /// </summary>
    /// <param name="index">vertex index</param>
    /// <returns>true if removal was successful</returns>
    public bool Remove(int index)
    {
        if (index < 0 || index >= this.nodes.Count)
            return false;

        this.indices.Remove(nodes[index]);
        this.nodes.RemoveAt(index);
        this.edges.RemoveAt(index);

        // Remove edges that point to this node
        foreach (var edgeList in this.edges)
        {
            edgeList.RemoveAll(edge => edge.ToIndex == index);
        }

        // Recompute indices mapping (maybe there's a better way to do this)
        for (var i = index; i < this.nodes.Count; i++)
        {
            var node = this.nodes[i];
            this.indices[node] = i;
        }

        // Do edges too because these could now be shifted by 1
        foreach (var edgelist in this.edges)
        {
            for (var j = 0; j < edgelist.Count; j++)
            {
                var edge = edgelist[j];
                int from = edge.FromIndex;
                int to = edge.ToIndex;

                if (from > index) from--;
                if (to > index) to--;

                edgelist[j] = new Edge(this, from, to, edge.Data);
            }
        }

        return true;
    }

    /// <summary>
    /// Remove all of the given vertices
    /// </summary>
    /// <param name="vertices">vertices to remove</param>
    /// <returns>true if vertices are removed</returns>
    public bool RemoveAll(params ReadOnlySpan<TVertex> vertices)
    {
        if (vertices.Length == 0)
            return false;

        bool removed = false;
        foreach (var vertex in vertices)
        {
            removed |= Remove(vertex);
        }

        return removed;
    }

    /// <summary>
    /// Replace a given vertex, edges are preserved
    /// </summary>
    /// <param name="original">original vertex value</param>
    /// <param name="replacement">replacement value</param>
    /// <returns>true if replacement was successful</returns>
    public bool Replace(TVertex replacement, TVertex original)
    {
        if (!this.indices.TryGetValue(original, out int index))
            return false; // Not in graph
        if (this.indices.ContainsKey(replacement))
            return false; // Already in graph (did the user intend to "swap"?)

        // Replace the vertex
        this.nodes[index] = replacement;
        this.indices.Remove(original);
        this.indices.Add(replacement, index);

        // Indices didn't change for anything else so we can just stop here

        return true;
    }

    /// <summary>
    /// Replace all vertices with a single replacement vertex, edges are preserved
    /// </summary>
    /// <param name="replacement">replacement value</param>
    /// <param name="originals">list of original vertices</param>
    /// <returns>true if replacement was successful</returns>
    public bool ReplaceAll(TVertex replacement, params ReadOnlySpan<TVertex> originals)
    {
        if (originals.Length == 0)
            return false; // Nothing to replace

        if (this.indices.ContainsKey(replacement))
            return false; // Already in graph (did the user intend to "swap"?)

        // Determine which nodes are to be removed
        HashSet<int> toRemove = new HashSet<int>();
        foreach (var vertex in originals)
        {
            if (indices.TryGetValue(vertex, out var index))
                toRemove.Add(index);
        }
        if (toRemove.Count == 0)
            return false; // Nothing removed

        // Determine indices
        Dictionary<int, int> old2new = new();
        int replacementIndex = 0;
        for (int i = 0; i < this.nodes.Count; i++)
        {
            if (toRemove.Contains(i))
                continue;

            old2new[i] = replacementIndex++;
        }
        foreach (var index in toRemove)
        {
            old2new[index] = replacementIndex;
        }

        // Accumulate outbound edges
        List<Edge> outbound = new();
        foreach (var i in toRemove)
        {
            foreach (var edge in edges[i])
            {
                if (toRemove.Contains(edge.ToIndex))
                    continue; // Skip self references

                if (old2new.TryGetValue(edge.ToIndex, out int mappedTo))
                {
                    outbound.Add(new Edge(this, replacementIndex, mappedTo, edge.Data));
                }
            }
        }

        // Delete nodes and edges
        foreach (var index in toRemove.OrderByDescending(x => x))
        {
            // Delete all nodes
            this.nodes.RemoveAt(index);
            // Delete all edges
            this.edges.RemoveAt(index);
        }

        // Add replacement node & it's edges
        this.nodes.Add(replacement);
        this.edges.Add(outbound);

        // Update edges
        foreach (var edgelist in this.edges)
        {
            for (var i = 0; i < edgelist.Count; i++)
            {
                var edge = edgelist[i];
                edgelist[i] = new Edge(this, old2new[edge.FromIndex], old2new[edge.ToIndex], edge.Data);
            }
        }

        // Update indices
        this.indices.Clear();
        for (var i = 0; i < this.nodes.Count; i++)
        {
            this.indices[this.nodes[i]] = i;
        }

        return true;
    }

    /// <summary>
    /// Test if the given vertex exists in the graph
    /// </summary>
    /// <param name="vertex">vertex</param>
    /// <returns>true if the vertex exists in the graph</returns>
    public bool ContainsVertex(TVertex vertex)
    {
        return this.indices.ContainsKey(vertex);
    }

    /// <summary>
    /// Test if the given directed edge exists
    /// </summary>
    /// <param name="from">starting vertex</param>
    /// <param name="to">ending vertex</param>
    /// <returns>true if there is an edge from the starting verted to the ending vertex</returns>
    public bool HasEdge(TVertex from, TVertex to)
    {
        if (!this.indices.TryGetValue(from, out var index) || !this.indices.TryGetValue(to, out var toIndex))
            return false;

        return this.edges[index].Any(edge => edge.ToIndex == toIndex);
    }

    /// <summary>
    /// Get the edge between two vertices if it exists
    /// </summary>
    /// <param name="from">starting vertex</param>
    /// <param name="to">ending vertex</param>
    /// <returns>edge or null</returns>
    public Edge? GetEdge(TVertex from, TVertex to)
    {
        if (!this.indices.TryGetValue(from, out var index) || !this.indices.TryGetValue(to, out var toIndex))
            return null;
        return this.edges[index].FirstOrDefault(edge => edge.ToIndex == toIndex);
    }

    /// <summary>
    /// Create a directed connection two vertices
    /// </summary>
    /// <param name="from">starting vertex</param>
    /// <param name="to">ending vertex</param>
    /// <param name="data">edge data</param>
    /// <returns>edge or null</returns>
    public Edge? Connect(TVertex from, TVertex to, TEdgeData? data = default(TEdgeData))
    {
        if (!this.indices.TryGetValue(from, out var fromIndex) || !this.indices.TryGetValue(to, out var toIndex))
            return null;

        var e = new Edge(this, fromIndex, toIndex, data);
        this.edges[indices[from]].Add(e);
        return e;
    }

    /// <summary>
    /// Disconnect two vertices
    /// </summary>
    /// <param name="from">starting vertex</param>
    /// <param name="to">ending vertex</param>
    /// <returns>true if disconnection was successful</returns>
    public bool Disconnect(TVertex from, TVertex to)
    {
        if (!indices.TryGetValue(from, out int index))
            return false;

        if (!indices.TryGetValue(to, out int toIndex))
            return false;

        int removed = this.edges[index].RemoveAll(edge => edge.ToIndex == toIndex);
        return removed > 0;
    }

    /// <summary>
    /// Compute the index of the vertex in the node list
    /// </summary>
    /// <param name="data">vertex</param>
    /// <returns>-1 or the index to the vertex</returns>
    public int IndexOf(TVertex data)
    {
        if (!this.indices.TryGetValue(data, out int index))
            return -1;
        return index;
    }

    /// <summary>
    /// Get a given vertex value by it's index
    /// </summary>
    /// <param name="index">vertex index</param>
    /// <returns>vertex value</returns>
    public TVertex GetVertex(int index) => this.nodes[index];

    /// <summary>
    /// All edges incomming to the given vertex
    /// </summary>
    /// <param name="to">vertex</param>
    /// <returns>edges incomming to the given vertex</returns>
    public IEnumerable<Edge> IncomingEdges(TVertex to)
    {
        if (!indices.TryGetValue(to, out int index))
            return Enumerable.Empty<Edge>();

        return this.edges.SelectMany((set) => set.Where(edge => edge.ToIndex == index));
    }

    /// <summary>
    /// All edges incomming to the given vertex
    /// </summary>
    /// <param name="to">vertex index</param>
    /// <returns>edges incomming to the given vertex</returns>
    public IEnumerable<Edge> IncomingEdges(int to)
    {
        if (to < 0 || to >= this.edges.Count)
            return Enumerable.Empty<Edge>();

        return this.edges.SelectMany((set) => set.Where(edge => edge.ToIndex == to));
    }

    /// <summary>
    /// All edges outgoing from the given vertex
    /// </summary>
    /// <param name="from">vertex</param>
    /// <returns>edges outgoing from the given vertex</returns>
    public IEnumerable<Edge> OutgoingEdges(TVertex from)
    {
        if (!this.indices.TryGetValue(from, out int index))
            return Enumerable.Empty<Edge>();

        return this.edges[index];
    }

    /// <summary>
    /// All edges outgoing from the given vertex
    /// </summary>
    /// <param name="from">vertex index</param>
    /// <returns>edges outgoing from the given vertex</returns>
    public IEnumerable<Edge> OutgoingEdges(int from)
    {
        if (from < 0 || from >= this.edges.Count)
            return Enumerable.Empty<Edge>();

        return this.edges[from];
    }

    /// <summary>
    /// Test if this graph contains any loops
    /// </summary>
    /// <returns>true if loops are present, false otherwise</returns>
    public bool IsCyclic()
    {
        int nodeCount = nodes.Count;
        bool[] visited = new bool[nodeCount];
        bool[] inStack = new bool[nodeCount];


        bool IsCyclicIterative(int startIndex, bool[] visited, bool[] inStack)
        {
            Stack<(int nodeIndex, IEnumerator<Edge> iterator)> stack = new Stack<(int, IEnumerator<Edge>)>();

            stack.Push((startIndex, edges[startIndex].GetEnumerator()));
            visited[startIndex] = true;
            inStack[startIndex] = true;

            while (stack.Count > 0)
            {
                var (current, iterator) = stack.Peek();

                if (!iterator.MoveNext())
                {
                    // Done processing current node
                    stack.Pop();
                    inStack[current] = false;
                    continue;
                }

                Edge edge = iterator.Current;
                int neighborIndex = edge.ToIndex;

                if (!visited[neighborIndex])
                {
                    visited[neighborIndex] = true;
                    inStack[neighborIndex] = true;
                    stack.Push((neighborIndex, edges[neighborIndex].GetEnumerator()));
                }
                else if (inStack[neighborIndex])
                {
                    // Found a node already in current DFS path → cycle
                    return true;
                }
            }

            return false;
        }

        for (int i = 0; i < nodeCount; i++)
        {
            if (!visited[i])
            {
                if (IsCyclicIterative(i, visited, inStack))
                    return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Compute a topological sorting of all vertices in this graph. The graph must be acylic for this to work.
    /// </summary>
    /// <returns>topological sorting</returns>
    /// <exception cref="Exception">thrown if no ordering could be created</exception>
    public List<TVertex> TopologicalSort()
    {
        int n = this.nodes.Count;
        bool[] visited = new bool[n];
        bool[] recursionStack = new bool[n];
        List<TVertex> ordered = new List<TVertex>(n);

        Stack<(int node, bool previsit)> stack = new Stack<(int, bool)>();
        for (int start = 0; start < n; start++)
        {
            if (visited[start])
                continue;

            stack.Clear();
            stack.Push((start, true));

            while (stack.Count > 0)
            {
                var (v, previsit) = stack.Pop();

                if (previsit)
                {
                    if (recursionStack[v])
                        throw new InvalidOperationException("Topological sorting is not possible on cyclic graphs");

                    if (visited[v])
                        continue;

                    recursionStack[v] = true;
                    stack.Push((v, false)); // Post-visit marker

                    var edges = this.edges[v];
                    foreach (var edge in edges)
                    {
                        var u = edge.ToIndex;
                        stack.Push((u, true)); // Visit children first
                    }
                }
                else
                {
                    if (!visited[v])
                    {
                        visited[v] = true;
                        recursionStack[v] = false;
                        ordered.Add(this.nodes[v]);
                    }
                }
            }
        }

        ordered.Reverse();
        return ordered;
    }

    /// <summary>
    /// Write the graph to an SVG file
    /// </summary>
    /// <param name="writer">writer to write the SVG to</param>
    /// <param name="layout">object which is used to layout graph nodes</param>
    /// <param name="vertexRenderer">renderer to draw graph nodes</param>
    /// <param name="edgeRenderer">renderer to draw graph edges</param>
    public void WriteSvg(TextWriter writer, IVertexLayout<TVertex, TEdgeData> layout, IVertexRenderer<TVertex> vertexRenderer, IEdgeRenderer<TEdgeData> edgeRenderer)
    {
        var vertices = this.nodes;

        // Layout graph (whatever strategy)
        layout.Layout(this, out int width, out int height, out List<RectangleF> bounds);

        // Begin SVG
        writer.WriteLine($"<svg width=\"{width}\" height=\"{height}\" xmlns=\"http://www.w3.org/2000/svg\">");
        writer.WriteLine("<defs>");
        writer.WriteLine("<marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"10\" refY=\"3\" orient=\"auto\" markerUnits=\"strokeWidth\">");
        writer.WriteLine("<path d=\"M0,0 L0,6 L9,3 z\" fill=\"black\" />");
        writer.WriteLine("</marker>");
        writer.WriteLine("</defs>");

        // Draw edges
        foreach (var edge in EnumerateEdges())
        {
            var from = bounds[edge.FromIndex];
            var to = bounds[edge.ToIndex];
            var fromCenter = new PointF((from.Right - from.Left) / 2.0f, (from.Top - from.Bottom) / 2.0f);
            var toCenter = new PointF((to.Right - to.Left) / 2.0f, (to.Top - to.Bottom) / 2.0f);

            edgeRenderer.Render(writer, edge.Data, fromCenter, toCenter);
        }

        // Draw vertices
        for (var i = 0; i < vertices.Count; i++)
        {
            var vertex = vertices[i];
            var bound = bounds[i];

            vertexRenderer.Render(writer, vertex, bound);
        }

        // End SVG
        writer.WriteLine("</svg>");
    }
}

public interface IVertexLayout<TVertex, TEdgeData>
where TVertex : notnull
{
    public void Layout(AdjacencyListGraph<TVertex, TEdgeData> graph, out int width, out int height, out List<RectangleF> positions);
}

/// <summary>
/// Layout that arranges all graph nodes using a force directed algorithm
/// </summary>
public class ForceDirectedLayout<TVertex, TEdgeData> : IVertexLayout<TVertex, TEdgeData>
where TVertex : notnull
{
    private readonly float nodeRadius;
    private readonly float nodeBuffer;
    private readonly float padding;
    private readonly int iterations;

    public ForceDirectedLayout(
        float nodeRadius,
        float nodeBuffer = 0f,
        float padding = 50f,
        int iterations = 500
    )
    {
        this.nodeRadius = nodeRadius;
        this.nodeBuffer = nodeBuffer;
        this.padding = padding;
        this.iterations = iterations;
    }

    private Random rng = new Random();

    public void Layout(AdjacencyListGraph<TVertex, TEdgeData> graph, out int width, out int height, out List<RectangleF> positions)
    {
        var vertexCount = graph.VertexCount;
        positions = new List<RectangleF>(vertexCount);
        if (vertexCount == 0)
        {
            width = height = 0;
            return;
        }

        // Intitial random placement (assume a 2D cartesian space and place around [0,0] try not to place them at the same spot)
        var bufferedRadius = nodeRadius + nodeBuffer;
        var diameter = bufferedRadius * 2;

        var placement_space = Math.Max(vertexCount * bufferedRadius, 100f); // Compute a space which we can randomly place our nodes

        for (var i = 0; i < vertexCount; i++)
        {
            var x = (float)((rng.NextDouble() - 0.5) * placement_space);
            var y = (float)((rng.NextDouble() - 0.5) * placement_space);
            positions[i] = new RectangleF(x - bufferedRadius, y - bufferedRadius, diameter, diameter);
        }

        // Iterative updates
        PointF[] displacements = new PointF[vertexCount];
        float area = (diameter) * (diameter) * vertexCount;
        float k = (float)Math.Sqrt(area / vertexCount);
        float temperature = k * 0.5f;

        for (var iteration = 0; iteration < this.iterations; iteration++)
        {
            Array.Clear(displacements, 0, vertexCount); // Reset displacements

            // Repulsive forces (nodes repel each other)
            for (int i = 0; i < vertexCount; i++)
            {
                for (int j = i + 1; j < vertexCount; j++)
                {
                    if (i == j)
                        continue;

                    var pi = GetCenter(positions[i]);
                    var pj = GetCenter(positions[j]);

                    var delta = Subtract(pi, pj);
                    float distance = MathF.Max(1f, Magnitude(delta)); // avoid divide-by-zero
                    float force = (k * k) / distance;

                    var direction = Normalize(delta);
                    var forceVec = Scale(direction, force);

                    displacements[i] = Add(displacements[i], forceVec);
                    displacements[j] = Subtract(displacements[j], forceVec); // Equal and opposite
                }
            }
            // Attractive forces (edges pull nodes together)
            foreach (var edge in graph.EnumerateEdges())
            {
                var i = edge.FromIndex;
                var j = edge.ToIndex;

                var pi = GetCenter(positions[i]);
                var pj = GetCenter(positions[j]);

                var delta = Subtract(pi, pj);
                float distance = MathF.Max(1f, Magnitude(delta));
                float force = (distance * distance) / k;

                var direction = Normalize(delta);
                var forceVec = Scale(direction, force);

                displacements[i] = Subtract(displacements[i], forceVec);
                displacements[j] = Add(displacements[j], forceVec);
            }

            // Apply with damping 
            for (int i = 0; i < vertexCount; i++)
            {
                var pos = GetCenter(positions[i]);
                var disp = displacements[i];

                float dispMag = Magnitude(disp);
                if (dispMag > temperature)
                {
                    disp = Scale(disp, temperature / dispMag); // limit movement
                }

                var newCenter = Add(pos, disp);

                // Re-center the rectangle at new position
                positions[i] = new RectangleF(
                    newCenter.X - bufferedRadius,
                    newCenter.Y - bufferedRadius,
                    diameter,
                    diameter
                );
            }

            // Cool temperature
            temperature *= 0.95f;
        }

        // Normalize positions into bounding box with padding
        float minX = positions.Min(r => r.Left);
        float minY = positions.Min(r => r.Top);
        float maxX = positions.Max(r => r.Right);
        float maxY = positions.Max(r => r.Bottom);

        float offsetX = padding - minX;
        float offsetY = padding - minY;

        for (int i = 0; i < vertexCount; i++)
        {
            var rect = positions[i];
            positions[i] = new RectangleF(
                rect.X + offsetX,
                rect.Y + offsetY,
                rect.Width,
                rect.Height
            );
        }

        width = (int)Math.Ceiling(maxX - minX + 2 * padding);
        height = (int)Math.Ceiling(maxY - minY + 2 * padding);
    }

    private static PointF Add(PointF a, PointF b) => new PointF(a.X + b.X, a.Y + b.Y);
    private static PointF Subtract(PointF a, PointF b) => new PointF(a.X - b.X, a.Y - b.Y);
    private static PointF Scale(PointF v, float factor) => new PointF(v.X * factor, v.Y * factor);
    private static float Magnitude(PointF v) => MathF.Sqrt(v.X * v.X + v.Y * v.Y);
    private static PointF Normalize(PointF v)
    {
        float mag = Magnitude(v);
        return mag > 1e-5f ? new PointF(v.X / mag, v.Y / mag) : new PointF(0, 0);
    }

    private static PointF GetCenter(RectangleF rect) => new PointF(rect.X + rect.Width / 2f, rect.Y + rect.Height / 2f);
}

/// <summary>
/// Layout which arranges all graph nodes in one large circle in increasing node order
/// </summary>
public class CircularLayout<TVertex, TEdgeData> : IVertexLayout<TVertex, TEdgeData>
where TVertex : notnull
{
    private readonly float radius;
    private readonly float buffered_radius;
    private readonly float padding;
    private readonly float startAngle;
    public CircularLayout(float nodeRadius, float nodeBuffer = 0, float padding = 50, float startAngle = 0)
    {
        this.radius = Math.Max(0, nodeRadius);
        this.buffered_radius = Math.Max(0, nodeRadius + nodeBuffer);
        this.padding = Math.Max(0, padding);
        this.startAngle = startAngle;
    }

    public void Layout(AdjacencyListGraph<TVertex, TEdgeData> graph, out int width, out int height, out List<RectangleF> positions)
    {
        var vertices = graph.VertexCount;
        positions = new List<RectangleF>(vertices);
        if (vertices == 0)
        {
            width = height = 0;
            return;
        }

        var diameter = 2 * buffered_radius;
        float radius = (vertices * diameter) / (2.0f * MathF.PI) + padding;
        float cx = radius + padding;
        float cy = radius + padding;

        width = (int)Math.Ceiling(2 * radius + 2 * padding);
        height = (int)Math.Ceiling(2 * radius + 2 * padding);

        for (int i = 0; i < vertices; i++)
        {
            double angle = startAngle + 2 * Math.PI * i / vertices;
            float x = cx + radius * MathF.Cos((float)angle);
            float y = cy + radius * MathF.Sin((float)angle);

            // Rectangle centered at (x, y)
            var rect = new RectangleF(
                x - buffered_radius,
                y - buffered_radius,
                diameter,
                diameter
            );

            positions.Add(rect);
        }
    }
}

public interface IVertexRenderer<TVertex>
{
    public void Render(TextWriter writer, TVertex? label, RectangleF region);
}

/// <summary>
/// Renderer that renders all graph nodes as a cicle
/// </summary>
public class CircleRenderer<TVertex> : IVertexRenderer<TVertex>
{
    private string style;

    public CircleRenderer() : this("fill: white; stroke: black; stroke-width: 5px;") { }
    public CircleRenderer(string fill, string stroke, int stroke_width) : this($"fill: {fill}; stroke: {stroke}; stroke-width: {stroke_width}px;") { }
    public CircleRenderer(string style)
    {
        this.style = style;
    }

    public void Render(TextWriter writer, TVertex? label, RectangleF region)
    {
        var center = new PointF((region.Right - region.Left) / 2.0f, (region.Top - region.Bottom) / 2.0f);
        var r = region.Width / 2.0f;
        writer.Write("<g>");
        writer.Write($"<circle r=\"{r}\" cx=\"{center.X}\" cy=\"{center.Y}\" style=\"{style}\"/>");
        /*if (!string.IsNullOrEmpty(label))
        {
            writer.Write($"<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" dominant-baseline=\"middle\">{System.Security.SecurityElement.Escape(label)}</text>");
        }*/
        writer.WriteLine("</g>");
    }
}

public interface IEdgeRenderer<TEdgeData>
{
    public void Render(TextWriter writer, TEdgeData? label, PointF from, PointF to);
}

/// <summary>
/// Renderer that rendered all graph edges with solid lines
/// </summary>
public class SolidEdge<TEdgeData> : IEdgeRenderer<TEdgeData>
{
    private string style;

    public SolidEdge() : this("stroke: black; stroke-width: 5px;") { }
    public SolidEdge(string stroke, int stroke_width) : this($"stroke: {stroke}; stroke-width: {stroke_width}px;") { }
    public SolidEdge(string style)
    {
        this.style = style;
    }

    public void Render(TextWriter writer, TEdgeData? label, PointF from, PointF to)
    {
        writer.Write("<g>");
        writer.Write($"<line x1=\"{from.X}\" y1=\"{from.Y}\" x2=\"{to.X}\" y2=\"{to.Y}\" style=\"{style}\" marker-end=\"url(#arrow)\"/>");
        /*if (!string.IsNullOrEmpty(label))
        {
            writer.Write($"<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" dominant-baseline=\"middle\">{System.Security.SecurityElement.Escape(label)}</text>");
        }*/
        writer.WriteLine("</g>");
    }
}