using System.Collections.ObjectModel;

namespace Qkmaxware.Terminal.Elements;

public class CharacterAnimation
{
    private TimeSpan interval;
    private List<char> frames;

    /// <summary>
    /// Interval between frames
    /// </summary>
    public TimeSpan Interval => interval;

    /// <summary>
    /// All characters that make up each frame
    /// </summary>
    public ReadOnlyCollection<char> Frames { get; private set; }

    public static CharacterAnimation TravellingDots => new CharacterAnimation(
            TimeSpan.FromMilliseconds(80),
            '⠋',
            '⠙',
            '⠹',
            '⠸',
            '⠼',
            '⠴',
            '⠦',
            '⠧',
            '⠇',
            '⠏'
        );
    public static CharacterAnimation MissingDot => new CharacterAnimation(
            TimeSpan.FromMilliseconds(80),
            '⣾',
			'⣽',
			'⣻',
			'⢿',
			'⡿',
			'⣟',
			'⣯',
			'⣷'
        );

    public CharacterAnimation(TimeSpan interval, params List<char> frames)
    {
        if (frames.Count == 0)
            throw new ArgumentException("At least 1 frame is required for a character animation");

        this.interval = interval;
        this.frames = frames;
        this.Frames = frames.AsReadOnly();
    }
}

public class Spinner: IElement
{
    public CharacterAnimation? Animation { get; set; }
    public string? Label { get; set; }

    public Spinner(CharacterAnimation anim, string? label = null)
    {
        this.Animation = anim;
        this.Label = label;
    }

    private int currentFrame = 0;
    private DateTime lastFrameTime = DateTime.MinValue;

    public LayoutSize Render(Graphics graphics)
    {
        if (Animation is null)
            return LayoutSize.Empty;

        DateTime now = DateTime.UtcNow;

        // Initialize on first render
        if (lastFrameTime == DateTime.MinValue)
        {
            lastFrameTime = now;
        }

        // Time to update the frame?
        if ((now - lastFrameTime) >= Animation.Interval)
        {
            currentFrame = (currentFrame + 1) % Animation.Frames.Count;
            lastFrameTime = now;
        }

        graphics.Draw(new System.Drawing.Point(0, 0), Animation.Frames[currentFrame]);

        int len = 0;
        if (this.Label is not null) {
            graphics.DrawClipped(new System.Drawing.Point(1, 0), this.Label);
            len = this.Label.Length;
        }

        return new LayoutSize(Math.Min(1 + len, graphics.DrawingRegion.Width), 1);
    }
}