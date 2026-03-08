
using System.Drawing;
using System.Text;
using DotML.Network;
using DotML.Network.Templates;
using SkiaSharp;

namespace DotML.Examples.Pong;

public interface IController
{
    public void Poll();

    public bool UpHeld();
    public bool DownHeld();
    public bool Accept();
}

public class HumanController : IController
{   
    private ConsoleKey upkey;
    private bool up;
    private ConsoleKey downkey;
    private bool down;
    private ConsoleKey acceptkey;
    private bool accept;

    public HumanController(bool isP1)
    {
        this.upkey = isP1 ? ConsoleKey.UpArrow : ConsoleKey.W;
        this.downkey = isP1 ? ConsoleKey.DownArrow : ConsoleKey.S;
        this.acceptkey = isP1 ? ConsoleKey.Spacebar: ConsoleKey.LeftWindows;
    }
    public bool Accept()
    {
        return this.accept;
    }

    public bool DownHeld()
    {
        return this.down;
    }

    public void Poll()
    {
        up = false;
        down = false;
        accept = false;

        if (Console.KeyAvailable)
        {
            var key = Console.ReadKey(false).Key;
            if (key == upkey)
                up = true;
            else if (key == downkey)
                down = true;
            else if (key == acceptkey)
                accept = true;
        }
    }

    public bool UpHeld()
    {
        return this.up;
    }
}

public class SimpleAiController: IController
{
    private Game game;
    private bool isP1;
    private bool up; private bool down;

    public SimpleAiController(Game game, bool isP1)
    {
        this.game = game;
        this.isP1 = isP1;
    }

    public bool Accept() => false;

    public bool DownHeld() => down;

    public void Poll()
    {
        var y = (isP1 ? game.Left.Y : game.Right.Y);
        var offset = game.Ball.Position.Y - y;

        up = offset < -2;
        down = offset > 2;
    }

    public bool UpHeld() => up;
}

public class DummyController : IController
{
    public bool Accept() => false;

    public bool DownHeld() => false;

    public void Poll() {}

    public bool UpHeld() => false;
}

public class NNAiController: IController
{
    private Game game;
    private bool isP1;
    public INetworkModule Network {get; init;}

    private bool up; private bool down;

    private static MultilayerPerceptronFactory factory = new MultilayerPerceptronFactory();

    private const int InputDX = 0;
    private const int InputVX = 1;
    private const int InputVY = 2;

    private const int OutputUp = 0;
    private const int OutputDown = 1;

    public NNAiController(Game game, bool isP1)
    {
        this.game = game;
        this.isP1 = isP1;
        this.Network = MakeNetwork();
    }

    public NNAiController(Game game, bool isP1, INetworkModule network)
    {
        this.game = game;
        this.isP1 = isP1;
        this.Network = network;
    }

    public static INetworkModule MakeNetwork()
    {
        return factory.Make(new MultilayerPerceptronFactory.BuildSettings(
            ActivationFunctions.Tanh,
            4, 5, 2
        ));
    }

    public void Poll()
    {
        Game.Bar self = isP1 ? game.Left : game.Right;
        Game.BallData ball = game.Ball;

        // Get relative position
        (float X, float Y) selfPosition = (self.X, self.Y);
        (float X, float Y) ballPosition = ball.Position;
        (float X, float Y) relativePosition = (
            isP1 
                ? ballPosition.X - selfPosition.X // Left paddle: distance is positive
                : selfPosition.X - ballPosition.X, // Right paddle: flip sign so it's positive not negative distance
            ballPosition.Y - selfPosition.Y
        );

        // Get relative velocity
        (float X, float Y) ballVelocity = ball.Velocity;
        (float X, float Y) relativeVelocity = (
            isP1 
                ? ballVelocity.X      // Left paddle: negative means toward
                : -ballVelocity.X,    // Right paddle: flip sign so toward = negative
            ballVelocity.Y
        );
        
        // Do it
        var output = Network.Forward(Tensor<float>.Vec([
            relativePosition.X / game.Width,                    // 0 to 1
            relativePosition.Y / game.Height,                   // 0 to 1
            relativeVelocity.X / Game.BallData.SpeedAmount,     // -1 to 1
            relativeVelocity.Y / Game.BallData.SpeedAmount      // -1 to 1
        ]));
        var btn = Vec<float>.Wrap(output.AsArray()).IndexOfMaxValue();
        
        up = btn == OutputUp;
        down = btn == OutputDown;
    }

    public bool UpHeld() => up;

    public bool DownHeld() => down;

    public bool Accept() => false;
}

public class Game
{
    private Rectangle region = new Rectangle();
    private StringBuilder buffer;

    public int Width => region.Width;
    public int Height => region.Height;

    public Game(int width = 40, int height = 80)
    {
        var loc = Console.GetCursorPosition();
        this.region = new Rectangle(loc.Left, loc.Top, width, height);
        this.buffer = new StringBuilder(width * (height + 1) + height);

        this.Left = new Bar(0);
        this.Right = new Bar(width - Bar.BarWidth);
    }

const string title = 
@" ____                   _ 
|  _ \ ___  _ __   __ _| |
| |_) / _ \| '_ \ / _` | |
|  __/ (_) | | | | (_| |_|
|_|   \___/|_| |_|\__, (_)
                  |___/   ";

    public void FlushBuffer()
    {
        Console.SetCursorPosition(this.region.Left, this.region.Top);
        foreach (var span in buffer.GetChunks())
        {
            Console.Write(span);
        }
    }

    private int menuOption = 0;
    public enum Mode
    {
        Menu, VsAI, VsHuman, AiOnly
    }
    public Mode LoopMenu(TimeSpan dt, IController p1, IController p2)
    {
        buffer.Clear();
        buffer.AppendLine(title);
        buffer.AppendLine();
        
        buffer.Append(menuOption == 0 ? '>' : ' ');  buffer.AppendLine("Human VS AI");
        buffer.Append(menuOption == 1 ? '>' : ' '); buffer.AppendLine("Human VS Human");
        buffer.Append(menuOption == 2 ? '>' : ' '); buffer.AppendLine("Ai VS Ai");

        p1.Poll();

        if (p1.UpHeld())
        {
            menuOption = Math.Clamp(0, menuOption - 1, 2);
        }
        else if (p1.DownHeld())
        {
            menuOption = Math.Clamp(0, menuOption + 1, 2);
        }
        if (p1.Accept()) {
            ResetGame();
            return menuOption switch
            {   
                0 => Mode.VsAI,
                1 => Mode.VsHuman,
                2 => Mode.AiOnly,
                _ => throw new NotImplementedException()
            };
        }

        return Mode.Menu;
    }

    public class Bar
    {
        public const int BarWidth = 1;
        public const int BarHeight = 6;
        public const int BarHalfHeight = BarHeight / 2;
        public float Speed = 8;
        public float Y;
        public float X {get; init;}
        public int Score;
        public Bar(float x)
        {
            this.X = x;
        }
        public RectangleF Rectangle => new RectangleF(
            x: this.X,
            y: this.Y - Bar.BarHalfHeight,
            width: Bar.BarWidth,
            height: Bar.BarHeight
        );
    }
    public Bar Left;
    public Bar Right;
    public class BallData
    {
        public const int Radius = 2;
        public (float X, float Y) Position;
        public (float X, float Y) Velocity;
        public const float SpeedAmount = 16;
        public float Speed => MathF.Sqrt(Velocity.X * Velocity.X + Velocity.Y * Velocity.Y);
        public RectangleF Rectangle => new RectangleF(
            x: Position.X - BallData.Radius,
            y: Position.Y - BallData.Radius,
            width: 2 * BallData.Radius,
            height: 2 * BallData.Radius
        );
    }
    public BallData Ball = new BallData();
    public TimeSpan gameTime = TimeSpan.Zero;
    public void ResetGame()
    {
        ResetPositions();
        Left.Score = 0;
        Right.Score = 0;
        gameTime = TimeSpan.Zero;
    } 
    public void ResetPositions(bool isDirLeft = false)
    {
        float up = Random.Shared.Next((int)BallData.SpeedAmount * 2) - BallData.SpeedAmount;

        Left.Y = this.region.Height / 2;
        Right.Y = this.region.Height / 2;
        Ball.Position = (this.region.Width / 2, this.region.Height / 2);
        Ball.Velocity = normalize((
            isDirLeft ? BallData.SpeedAmount : -BallData.SpeedAmount,
            up
        ));
        
    }
    public struct FrameEvents
    {
        public bool LeftReturned = false;
        public bool RightReturned = false;
        public bool LeftScored = false;
        public bool RightScored = false;
        public float MissedItBy = 0;

        public FrameEvents() { }
    }
    public void LoopGame(TimeSpan dt, IController p1, IController p2, out FrameEvents events)
    {
        float dtSeconds = (float)dt.TotalSeconds;
        gameTime += dt;
        FrameEvents data = new();

        // User Input
        p1.Poll(); 
        var p1Dir = (p1.UpHeld() ? -1.0f : 0.0f) + (p1.DownHeld() ? 1.0f : 0.0f);
        p2.Poll();
        var p2Dir = (p2.UpHeld() ? -1.0f : 0.0f) + (p2.DownHeld() ? 1.0f : 0.0f);

        float p1Velocity = p1Dir * Left.Speed * dtSeconds;
        float p2Velocity = p2Dir * Right.Speed * dtSeconds;

        // Paddle Motion (simple)
        Left.Y = Math.Clamp(
            Left.Y + p1Velocity,
            Bar.BarHalfHeight,
            region.Height - Bar.BarHalfHeight);

        Right.Y = Math.Clamp(
            Right.Y + p2Velocity,
            Bar.BarHalfHeight,
            region.Height - Bar.BarHalfHeight);

        // Ball Motion
        Ball.Position = (
            Ball.Position.X + Ball.Velocity.X * dtSeconds,
            Ball.Position.Y + Ball.Velocity.Y * dtSeconds);

        // Wall Bounce
        if (Ball.Position.Y < BallData.Radius)
        {
            Ball.Position.Y = BallData.Radius;
            Ball.Velocity.Y = Math.Abs(Ball.Velocity.Y);
        }
        else if (Ball.Position.Y > region.Height - BallData.Radius)
        {
            Ball.Position.Y = region.Height - BallData.Radius;
            Ball.Velocity.Y = -Math.Abs(Ball.Velocity.Y);
        }

        RectangleF ballRect = Ball.Rectangle;
        RectangleF leftBar = Left.Rectangle;
        RectangleF rightBar = Right.Rectangle;

        // Paddle Collision (with spin)
        const float spinFactor = 0.35f;

        if (ballRect.IntersectsWith(leftBar) && Ball.Velocity.X < 0)
        {
            float offset = (Ball.Position.Y - Left.Y) / Bar.BarHalfHeight; // -1..1
            Ball.Velocity.X = Math.Abs(Ball.Velocity.X);
            Ball.Velocity.Y += offset * BallData.SpeedAmount * 0.5f;

            // Add spin from paddle motion
            Ball.Velocity.Y += p1Velocity * spinFactor;
            data.LeftReturned = true;
        }
        else if (ballRect.IntersectsWith(rightBar) && Ball.Velocity.X > 0)
        {
            float offset = (Ball.Position.Y - Right.Y) / Bar.BarHalfHeight; // -1..1
            Ball.Velocity.X = -Math.Abs(Ball.Velocity.X);
            Ball.Velocity.Y += offset * BallData.SpeedAmount * 0.5f;

            // Add spin from paddle motion
            Ball.Velocity.Y += p2Velocity * spinFactor;
            data.RightReturned = true;
        }

        // Normalize Ball Speed
        Ball.Velocity = normalize(Ball.Velocity);

        // Scoring
        if (Ball.Position.X < BallData.Radius && Ball.Velocity.X < 0)
        {
            Right.Score++;
            ResetPositions(isDirLeft: true);
            
            data.RightScored = true;
            data.MissedItBy = Math.Abs(Right.Y - Ball.Position.Y) - Bar.BarHalfHeight;
        }
        else if (Ball.Position.X > region.Width - BallData.Radius && Ball.Velocity.X > 0)
        {
            Left.Score++;
            ResetPositions(isDirLeft: false);

            data.LeftScored = true;
            data.MissedItBy = Math.Abs(Left.Y - Ball.Position.Y) - Bar.BarHalfHeight;
        }

        // Render
        var buffer = this.buffer;
        buffer.Clear();

        int centerX = region.Width / 2;

        for (int r = 0; r < region.Height; r++)
        {
            for (int c = 0; c < region.Width; c++)
            {
                var point = new PointF(c, r);

                // Paddles
                if (leftBar.Contains(point) || rightBar.Contains(point))
                {
                    buffer.Append('#');
                }
                // Ball (circle)
                else if (ballRect.Contains(point))
                {
                    float dx = point.X - Ball.Position.X;
                    float dy = point.Y - Ball.Position.Y;
                    float sqr = dx * dx + dy * dy;

                    buffer.Append(sqr <= BallData.Radius * BallData.Radius ? '*' : ' ');
                }
                // Center dashed line
                else if (c == centerX && r % 2 == 0)
                {
                    buffer.Append('|');
                }
                else
                {
                    buffer.Append(' ');
                }
            }

            buffer.AppendLine();
        }

        // Score Rendering (vertical on center line)
        string scoreText = $"{Left.Score}-{Right.Score}";
        int scoreStartCol = Math.Max(0, region.Width / 2 - scoreText.Length / 2);

        for (int i = 0; i < scoreText.Length; i++)
        {
            var col = scoreStartCol + i;
            if (col >= 0 && col < region.Width)
            {
                buffer[col] = scoreText[i];
            }
        }

        events = data;
    }
    private static (float X, float Y) normalize((float X, float Y) vec)
    {
        float len = MathF.Sqrt(vec.X * vec.X +
                           vec.Y * vec.Y);

        if (len > 0.001f)
        {
            vec.X = (vec.X / len) * BallData.SpeedAmount;
            vec.Y = (vec.Y / len) * BallData.SpeedAmount;
        }

        return (vec.X, vec.Y);
    }
}