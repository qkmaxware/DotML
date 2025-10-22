namespace Qkmaxware.Terminal.Layout;

public class Conditional: IElement
{
    public Func<bool>? Condition { get; set; }
    public IElement? ChildComponent { get; set; }

    public Conditional() { }
    
    public Conditional(Func<bool> condition, IElement? child)
    {
        this.Condition = condition;
        this.ChildComponent = child;
    }

    public LayoutSize Render(Graphics graphics)
    {
        if (!(Condition?.Invoke() ?? false))
        {
            return LayoutSize.Empty;
        }

        return ChildComponent?.Render(graphics) ?? LayoutSize.Empty;
    }
}