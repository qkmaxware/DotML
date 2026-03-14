namespace DotML.Cli.Visualizations;

public class ProgressAction<T> : IProgress<T>
{
    private Action<T> func;

    public ProgressAction(Action<T> func)
    {
        this.func = func;
    }

    public void Report(T value) => func(value);
}