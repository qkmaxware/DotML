using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Layout;
using Qkmaxware.Terminal.Elements;

class Person
{
    public string? Name { get; set; }
    public int Age { get; set; }
}

public class Program : ConsoleApp
{
    public static void Main()
    {
        var program = new Program();
        program.RenderLoop();
    }

    public Program()
    {
        this.Root = new VBox(
            new Panel(
                "Info",
                new Columns(
                    new Label(DateTime.Now.ToString()),
                    new Label("UserName"),
                    new Label(Environment.OSVersion.Platform.ToString())
                )
            ),
            new Panel(
                "Progress",
                new StaticProgressBar(0.5f)
            ),
            new Table<Person>(GeneratePeople()),
            new Panel(
                "Likelihood",
                new Padding(
                    new FixedLikelihood(new float[] { 0.1f, 0.4f, 0.5f }, new string[] { "Cat", "Cow", "Dog" }),
                    1, 1, 1, 1
                )
            ),
            new Panel(
                "2D Plot",
                new Padding(
                    new Plot2D(
                        title: "Trig Functions",
                        height: 12,
                        xLabel: "x",
                        yLabel: "y",
                        data: GenerateTrigData()
                    ),
                    1, 1, 1, 1
                )
            )
        );
    }
    
    private List<Series2D> GenerateTrigData()
    {
        var sin = new Series2D("sin(x)", symbol: Series2D.SymbolAsterisk);
        var cos = new Series2D("cos(x)", symbol: Series2D.SymbolPlus);

        var max = float.MinValue;
        var min = float.MaxValue;

        for (var x = -6.0f; x <= 6.0f; x += 0.1f)
        {
            var sx = MathF.Sin(x);
            var cx = MathF.Cos(x);
            sin.Add(x, sx);
            cos.Add(x, cx);
            max = Math.Max(max, Math.Max(sx, cx));
            min = Math.Min(min, Math.Min(sx, cx));
        }

        return new List<Series2D> { sin, cos };
    }

    private List<Person> GeneratePeople()
    {
        var people = new List<Person>
        {
            new Person {Name= "John Smith", Age = 24},
            new Person {Name= "Jane Doe", Age= 26}
        };
        return people;
    }

    public override void OnKey(ConsoleKeyInfo key)
    {
        if (key.Key == ConsoleKey.Escape)
            Environment.Exit(exitCode: 0);
    }

    public override void AfterRender()
    {
        // Halt refresh until a button is pressed
        Console.Read();
    }
}
