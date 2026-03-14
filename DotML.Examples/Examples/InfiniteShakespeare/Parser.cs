using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace DotML.Examples.InfiniteShakespeare;

#region AST
public class Play
{
    public string? Name {get; set;}
    public HashSet<string> Characters {get;} = new(StringComparer.InvariantCultureIgnoreCase);
    public List<Act> Acts {get;} = new();

    public void WriteTo(TextWriter writer, int indent = 0)
    {
        var indentStr = new string(' ', indent);
        writer.Write(indentStr);writer.WriteLine('{');
        writer.Write(indentStr);writer.Write("  Characters: [");
        writer.Write(string.Join(", ", Characters));
        writer.WriteLine("]");
        writer.Write(indentStr); writer.WriteLine("  Acts: [");
        foreach (var act in Acts)
        {
            act.WriteTo(writer, indent + 4);
        }
        writer.Write(indentStr);writer.WriteLine("  ]");
        writer.Write(indentStr);writer.WriteLine('}');
    }
}

public class Act
{
    public int Number {get; set;}
    public List<Scene> Scenes {get;} = new();

    public void WriteTo(TextWriter writer, int indent = 0)
    {
        var indentStr = new string(' ', indent);
        writer.Write(indentStr);writer.WriteLine('{');
        writer.Write(indentStr);writer.WriteLine($"  Number: {Number}");
        writer.Write(indentStr); writer.WriteLine("  Scenes: [");
        foreach (var scene in Scenes)
        {
            scene.WriteTo(writer, indent + 4);
        }
        writer.Write(indentStr);writer.WriteLine("  ]");
        writer.Write(indentStr);writer.WriteLine('}');
    }
}

public class Scene
{
    public int Number {get; set;}
    public string? Dialog {get; set;}

    public void WriteTo(TextWriter writer, int indent = 0)
    {
        var indentStr = new string(' ', indent);
        writer.Write(indentStr);writer.WriteLine('{');
        writer.Write(indentStr);writer.WriteLine($"  Number: {Number}");
        writer.Write(indentStr);writer.Write("  Dialog: "); writer.WriteLine(JsonSerializer.Serialize(Dialog));
        writer.Write(indentStr);writer.WriteLine('}');
    }
}

#endregion

public class Parser
{
    private static Regex _characterRegex = new(@"[A-Z][A-Z\s]+", RegexOptions.Compiled);
    private static Regex _actRegex = new(@"^ACT (?<num>\d+)$", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _sceneRegex = new(@"^Scene (?<num>\d+)$", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _dividerRegex = new(@"^[-_=]+$", RegexOptions.Compiled);

    /// <summary>
    /// Parse a Shakespeare play from a TextReader
    /// </summary>
    /// <param name="reader">reader to a .txt file</param>
    /// <returns>structured Play object</returns>
    public Play Parse(TextReader reader)
    {
        var play = new Play();
        string? line;

        // Header
        var name = reader.ReadLine();
        play.Name = name;
        while ((line = reader.ReadLine()) != "Characters in the Play") { /*Consume the rest of the header*/ }

        // Read all characters (find all CAPS names (can contain spaces))
        reader.ReadLine(); // Skip "----------------------"
        while (!string.IsNullOrEmpty((line = reader.ReadLine())))
        {
            foreach (Match match in _characterRegex.Matches(line))
            {
                play.Characters.Add(match.Value.Trim());
            }
        }

        // Dialog(s)
        Act? current_act = null;
        Scene? current_scene = null;
        StringBuilder currentDialog = new();

        void try_flush_dialog()
        {
            if (currentDialog.Length > 0 && current_scene is not null)
                {
                    var dialogText = currentDialog.ToString();
                    current_scene.Dialog = dialogText;
                    currentDialog.Clear();
                }
        }
        while ((line = reader.ReadLine()) != null)
        { 
            line = line.Trim();
            if (string.IsNullOrEmpty(line))
            {
                try_flush_dialog();
                continue;
            }

            // Act?
            var actMatch = _actRegex.Match(line);
            if (actMatch.Success)
            {
                try_flush_dialog();
                var act = new Act { Number = int.Parse(actMatch.Groups["num"].Value) };
                play.Acts.Add(act);
                current_act = act;
                continue;
            }

            // Scene?
            var sceneMatch = _sceneRegex.Match(line);
            if (sceneMatch.Success)
            {
                try_flush_dialog();
                if (current_act is null) {
                    current_act = new Act { Number = 1 };
                    play.Acts.Add(current_act);
                }
                var scene = new Scene { Number = int.Parse(sceneMatch.Groups["num"].Value) };
                current_act.Scenes.Add(scene);
                current_scene = scene;
                continue;
            }

            // Divider?
            if (_dividerRegex.IsMatch(line))
            {
                try_flush_dialog();
                continue;
            }

            // Dialog Line
            // Can be speech or stage direction
            // Stage directions start with [ and end with ]
            // Speech and stage direction can span multiple lines until a blank line or one of the other line types (act, scene, divider, character, new direction etc)
            currentDialog.AppendLine(line);
        }

        return play;
    }
}