using System.Diagnostics.CodeAnalysis;
using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

namespace DotML.Cli;

public abstract class BaseCommand {

    public static readonly int OKAY = 0;
    public static readonly int GENERIC_ERROR = 1;

    public abstract void Action(AppData appData);

    public int TryDoAction(AppData appData)
    {
        Console.Title = "DotML.NetFlow";
        try
        {
            Console.WriteLine();
            Action(appData);
            Console.WriteLine();
            return OKAY;
        }
        catch (ErrorCodeException ex)
        {
            return ex.ErrorCode;
        }
        catch (Exception e)
        {
            DrawDivider();
            Console.WriteLine("An unexpected error occurred:");
            Console.WriteLine(e);
            return GENERIC_ERROR;
        }
    }

    private static string[] truthy = [
        // English
        "true", "yes", "y", 
        // French
        "oui", 
        // Spanish, Italian
        "si", 
        // German, Dutch, Danish, Norwegian 
        "ja", 
        // Finnish
        "kyllä",
        // Japanese
        "はい", "hai",
    ];
    protected static bool IsSet(string? value)
    {
        return
            !string.IsNullOrEmpty(value)    // Not empty
            && truthy.Where(v => v.Equals(value, StringComparison.CurrentCultureIgnoreCase)).Any() // And is a truth string
        ;
    }

    protected class ErrorApp : ConsoleApp
    {
        public ErrorApp(string msg)
        {
            this.Root = new Panel("Error", new Label(msg)).WithPadding(1);
        }
        public ErrorApp(string msg, string[] options)
        {
            this.Root = new Panel(
                "Error",
                new VBox(
                    new Paragraph(msg),
                    new Label(string.Empty),
                    new Label("Options:"),
                    new UnorderedList(options.Select(x => new Label(x)))
                )
            ).WithPadding(1);
        }
    }
    protected class RenderView : ConsoleApp
    {
        public RenderView(IElement root) : base(root) { }
    }
    
    protected enum TaskState
    {
        Waiting, Running, Done
    }

    protected IElement MakeTaskView(Func<TaskState> condition, string name)
    {
        return new VBox(
            new Conditional(
                () => condition() == TaskState.Waiting,
                new Label(name)
            ),
            new Conditional(
                () => condition() == TaskState.Running,
                new Spinner(CharacterAnimation.TravellingDots, name)
            ),
            new Conditional(
                () => condition() == TaskState.Done,
                new Label("✓" + name)
            )
        );
    }

    protected IElement MakeTaskView(Func<TaskState> condition, Func<float> progress, string name)
    {
        return new VBox(
            new Conditional(
                () => condition() == TaskState.Waiting,
                new Label(name)
            ),
            new Conditional(
                () => condition() == TaskState.Running,
                new VSplitContainer( 
                    name.Length + 1, // First component takes up this much space, second component the rest
                    new Spinner(CharacterAnimation.TravellingDots, name),
                    new DynamicProgressBar(progress)
                )
            ),
            new Conditional(
                () => condition() == TaskState.Done,
                new Label("✓" + name)
            )
        );
    }

    protected void WriteError(string message)
    {
        new ErrorApp(message).RenderOnce();
        return;
    }

    protected void WriteError(string message, IEnumerable<string> options)
    {
        new ErrorApp(message, options.ToArray()).RenderOnce();
        return;
    }

    protected void DrawDivider(int? size = null) {
        Console.WriteLine();
        Console.WriteLine(new string('-', size.HasValue ? Math.Max(0, size.Value) : (Console.WindowWidth - 1)));
        Console.WriteLine();
    }

    protected string ColumnValue(object? obj, int colLength) {
        var str = Truncate(obj?.ToString() ?? string.Empty, colLength);
        return str.PadRight(colLength, ' ');
    }

    protected string Truncate(string str, int maxLength) {
        if (str.Length <= maxLength) {
            return str;
        }
        
        // IE "some long string" becomes "some long str..."
        return str.Substring(0, maxLength - 3) + "...";
    }

    public IEnumerable<string> EnumerateEmbeddedDocs() {
        var assembly = typeof(BaseCommand).Assembly;
        foreach (var file in assembly.GetManifestResourceNames()) {
            yield return file;
        }
    }

    public bool TryGetEmbeddedDoc(string filename, [NotNullWhen(true)] out string? contents) {
        var assembly = typeof(BaseCommand).Assembly;
        var resource = "DotML.Cli." + filename.Replace(' ', '_').Replace('\\', '.').Replace('/', '.');
        contents = null;
        try {
            using(var stream = assembly.GetManifestResourceStream(resource)) {
                if (stream is null)
                    return false;

                using(var reader = new StreamReader(stream)) {
                    contents = reader.ReadToEnd();
                    return true;
                }
            }
        } catch {
            return false;
        }
    }
}
public class ErrorCodeException : System.Exception
{
    public int ErrorCode { get; private set; }
    public ErrorCodeException(int error, string? message) : base(message)
    {
        this.ErrorCode = error;
    }
    public ErrorCodeException(int error, string? message, Exception? inner) : base(message, inner) { }
}