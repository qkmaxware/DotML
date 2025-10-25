using CommandLine;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.IO.Netbuild;
using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

namespace DotML.Cli.Commands;

[Verb("build", HelpText = "Compile a netbuild script into a network model")]
public class Build : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "File path to model to build", Required = true)]
    public string? FilePath {get; set;}

    [Option("description", Required = false, HelpText = "Text to use as the model's description")]
    public string? DescriptionText {get; set;}

    [Option("tag", HelpText = "Tag to use to uniquely identify this network once built", Required = false)]
    public IEnumerable<string>? TagsToAdd {get; set;}

    [Option("labels", Required = false, HelpText = "Labels for all output classes", Separator = ' ')]
    public IEnumerable<string>? LabelsForClasses {get; set;}

    public override void Action(AppData appData)
    {
        var view_stack = new VBox();
        var view = new RenderView(view_stack);
        string? error_string = null;
        var error_message = new Paragraph(() => error_string ?? string.Empty);
        var error_option_list = new List<IElement>();
        var error_panel = new Panel("Error", new VBox(
            error_message,
            new Conditional(() => error_option_list.Count > 0,
                new VBox(
                    new Label("Options:"),
                    new UnorderedList(error_option_list)
                )
            )
        )).WithPadding(1);
        view_stack.Add(new Conditional(() => !string.IsNullOrEmpty(error_string), error_panel));

        var success_view = new VBox();
        view_stack.Add(new Conditional(() => string.IsNullOrEmpty(error_string), success_view));

        var progress = new VBox();
        var progress_panel = new Panel("Progress", progress);

        var process_list = new UnorderedList();
        var process_panel = new Panel("Process", process_list).WithPadding(1);
        success_view.Add(process_panel);

        var fileReadTask = TaskState.Waiting;
        var fileReadTaskUi = MakeTaskView(() => fileReadTask, "Reading architecture file");
        process_list.Add(fileReadTaskUi);

        var modelLoadTask = TaskState.Waiting;
        var modelLoadTaskUi = MakeTaskView(() => modelLoadTask, "Compiling network module");
        process_list.Add(modelLoadTaskUi);

        var modelVerifyTask = TaskState.Waiting;
        var modelVerifyTaskUi = MakeTaskView(() => modelVerifyTask, "Validating model");
        process_list.Add(modelVerifyTaskUi);

        var metaTask = TaskState.Waiting;
        var metaTaskUi = MakeTaskView(() => metaTask, "Constructing metadata");
        process_list.Add(metaTaskUi);

        var savingTask = TaskState.Waiting;
        var savingTaskUi = MakeTaskView(() => savingTask, "Saving model");
        process_list.Add(savingTaskUi);

        var output_content_area = new Qkmaxware.Terminal.Layout.Padding(null, 1, 1, 1, 1);
        var output_panel = new Panel("Output", output_content_area);
        success_view.Add(new Conditional(() => output_content_area.ChildComponent is not null, output_panel));

        // Start the actual task
        var task = Task.Run(() => DoTasks(appData, ref error_string, ref fileReadTask, ref modelLoadTask, ref modelVerifyTask, ref metaTask, ref savingTask, output_content_area));

        // Wait (while rendering progress)
        var (left, top) = Console.GetCursorPosition();
        view.RenderWhile((self) => !task.IsCompleted);

        // Final render 
        if (task.IsFaulted)
        {
            error_string = task.Exception.Message;
            error_option_list.Clear();
        }
        Console.SetCursorPosition(left, top);
        view.RenderOnce();
    }

    private void DoTasks(AppData appData, ref string? error, ref TaskState fileLoad, ref TaskState modelLoad, ref TaskState modelVerify, ref TaskState metadataGeneration, ref TaskState fileCopying, Qkmaxware.Terminal.Layout.Padding output)
    {
        fileLoad = TaskState.Running;
        FileInfo file;
        if (string.IsNullOrEmpty(FilePath) || !((file = new FileInfo(FilePath)).Exists))
        {
            error = ($"The file '{FilePath}' doesn't exist.");
            return;
        }
        var contents = System.IO.File.ReadAllText(file.FullName);
        var format = file.Extension;
        fileLoad = TaskState.Done;

        modelLoad = TaskState.Running;
        var network = ModuleParser.Parse(contents, format);
        modelLoad = TaskState.Done;

        modelVerify = TaskState.Running;
        // TODO more verification tasks besides does it build
        modelVerify = TaskState.Done;

        metadataGeneration = TaskState.Running;
        var guid = Guid.NewGuid().ToString();
        var name = TagsToAdd?.FirstOrDefault() ?? (network is ArchitectureBlock arch ? arch.Name : null);
        var metaPath = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + ".xml"));
        var info = new ModelInfo(metaPath);
        info.Description = DescriptionText;
        if (TagsToAdd is not null)
        {
            info.Tags = new List<string>();
            foreach (var tag in TagsToAdd)
            {
                info.Tags.Add(tag);
            }
            if (!string.IsNullOrEmpty(name) && !info.Tags.Contains(name))
            {
                info.Tags.Add(name);
            }
        }
        if (LabelsForClasses is not null)
        {
            info.ProblemDescription = info.ProblemDescription ?? new ModelProblemDescription();
            info.ProblemDescription.Classification = new ModelProblemDescription.ClassificationDescription();
            var classLabels = new List<string>();
            foreach (var label in LabelsForClasses)
            {
                classLabels.Add(label);
            }
            info.ProblemDescription.Classification.ClassLabels = classLabels;
        }
        metadataGeneration = TaskState.Done;

        fileCopying = TaskState.Running;
        var archPath = new FileInfo(Path.Combine(appData.ModelDirectory.FullName, guid + format));
        using (var writer = new StreamWriter(archPath.FullName))
        {
            writer.Write(contents);
        }

        using (var writer = new StreamWriter(metaPath.FullName))
        {
            writer.Write(info.ToXml());
        }
        fileCopying = TaskState.Done;

        // TODO output panel
        var outBox = new VBox();
        outBox.Add(new Label($"Successfully built model {guid}."));
        if (name is not null)
            outBox.Add(new Label($"Successfully tagged model {guid} as '{name}'"));
        output.ChildComponent = outBox;
    }
}