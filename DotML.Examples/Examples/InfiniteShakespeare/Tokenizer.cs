using System.Text;
using DotML.Network.Embedding;

namespace DotML.Examples.InfiniteShakespeare;

#region Tokens

public enum ShakespeareTokenType
{
    Missing = 0,
    ActChange = 1,
    SceneChange = 2,
    SpeakerChange = 3,
    BeginStageDirection = 7,
    EndStageDirection = 8,
    Word = 4,             
    PausePunctuation = 5,       // e.g., comma, semicolon
    TerminalPunctuation = 6,    // e.g., period, exclamation mark, question mark
}

public abstract class ShakespeareToken: IEquatable<ShakespeareToken>
{
    protected ShakespeareTokenType TypeCode {get; private set;}
    public ShakespeareToken(ShakespeareTokenType typecode)
    {
        this.TypeCode = typecode;
    }

    public override bool Equals(object? obj)
    {
        return obj is ShakespeareToken token && Equals(token);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(TypeCode);
    }

    public abstract string ToParsableString();
    public abstract string ToWrittenString();

    public abstract bool Equals(ShakespeareToken? other);
}

public class MissingWordToken : ShakespeareToken
{
    private static MissingWordToken? _instance;
    public static MissingWordToken Instance => _instance ??= new MissingWordToken();
    private MissingWordToken(): base(ShakespeareTokenType.Missing) { }

    public override bool Equals(ShakespeareToken? other)
    {
        return other is MissingWordToken otherTok && TypeCode == otherTok.TypeCode;
    }

    
    public override string ToParsableString() => GetType().Name + "()";
    public override string ToWrittenString() => string.Empty;
}

public class ActChangeToken : ShakespeareToken
{
    private static ActChangeToken? _instance;
    public static ActChangeToken Instance => _instance ??= new ActChangeToken();
    private ActChangeToken(): base(ShakespeareTokenType.ActChange) { }

    public override bool Equals(ShakespeareToken? other)
    {
        return other is ActChangeToken otherTok && TypeCode == otherTok.TypeCode;
    }

    public override string ToParsableString() => GetType().Name + "()";
    public override string ToWrittenString() => Environment.NewLine + Environment.NewLine + "NEXT ACT" + Environment.NewLine + "==========" + Environment.NewLine + Environment.NewLine;
}

public class SceneChangeToken : ShakespeareToken
{
    private static SceneChangeToken? _instance;
    public static SceneChangeToken Instance => _instance ??= new SceneChangeToken();
    private SceneChangeToken(): base(ShakespeareTokenType.SceneChange) { }

    public override bool Equals(ShakespeareToken? other)
    {
        return other is SceneChangeToken otherTok && TypeCode == otherTok.TypeCode;
    }

    public override string ToParsableString() => GetType().Name + "()";
    public override string ToWrittenString() => Environment.NewLine + "NEXT SCENE" + Environment.NewLine + Environment.NewLine;
}

public class StageDirectionStart : ShakespeareToken
{
    private static StageDirectionStart? _instance;
    public static StageDirectionStart Instance => _instance ??= new StageDirectionStart();
    private StageDirectionStart(): base(ShakespeareTokenType.BeginStageDirection) { }

    public override bool Equals(ShakespeareToken? other)
    {
        return other is StageDirectionStart otherTok && TypeCode == otherTok.TypeCode;
    }

    public override string ToParsableString() => GetType().Name + "()";
    public override string ToWrittenString() => " [";
}

public class StageDirectionEnd : ShakespeareToken
{
    private static StageDirectionEnd? _instance;
    public static StageDirectionEnd Instance => _instance ??= new StageDirectionEnd();
    private StageDirectionEnd(): base(ShakespeareTokenType.EndStageDirection) { }

    public override bool Equals(ShakespeareToken? other)
    {
        return other is StageDirectionEnd otherTok && TypeCode == otherTok.TypeCode;
    }

    public override string ToParsableString() => GetType().Name + "()";
    public override string ToWrittenString() => " ]";
}

public class SpeakerChangeToken: ShakespeareToken
{
    public string Speaker { get; }
    public SpeakerChangeToken(string speaker) : base(ShakespeareTokenType.SpeakerChange)
    {
        Speaker = speaker.ToLowerInvariant();
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(TypeCode, Speaker);
    }

    public override bool Equals(ShakespeareToken? other)
    {
        return other is SpeakerChangeToken otherTok && TypeCode == otherTok.TypeCode && Speaker.Equals(otherTok.Speaker, StringComparison.InvariantCultureIgnoreCase);
    }

    public override string ToParsableString() => GetType().Name + "(" + Speaker + ")";
    public override string ToWrittenString() => Environment.NewLine + Speaker.ToUpperInvariant() + ":" + Environment.NewLine;
}

public class SpokenWord: ShakespeareToken
{
    public string Word { get; }
    public SpokenWord(string word) : base(ShakespeareTokenType.Word)
    {
        Word = word.ToLowerInvariant();
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(TypeCode, Word);
    }

    public override bool Equals(ShakespeareToken? other)
    {
        return other is SpokenWord otherTok && TypeCode == otherTok.TypeCode && Word.Equals(otherTok.Word, StringComparison.InvariantCultureIgnoreCase);
    }

    public override string ToParsableString() => GetType().Name + "(" + Word + ")";
    public override string ToWrittenString() => " " + Word;
}

public abstract class Punctuation: ShakespeareToken
{
    public char Symbol { get; }
    public Punctuation(ShakespeareTokenType type, char sym) : base(type)
    {
        this.Symbol = sym;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(TypeCode, Symbol);
    }

    public override bool Equals(ShakespeareToken? other)
    {
        return other is Punctuation otherTok && TypeCode == otherTok.TypeCode && Symbol == otherTok.Symbol;
    }
}

public class PausePunctuation : Punctuation
{
    public PausePunctuation(char sym) : base(ShakespeareTokenType.PausePunctuation, sym) { }

    public override string ToParsableString() => GetType().Name + "(" + Symbol.ToString() + ")";
    public override string ToWrittenString() => Symbol.ToString();
}

public class TerminalPunctuation : Punctuation
{
    public TerminalPunctuation(char sym) : base(ShakespeareTokenType.TerminalPunctuation, sym) { }

    public override string ToParsableString() => GetType().Name + "(" + Symbol.ToString() + ")";
    public override string ToWrittenString() => Symbol.ToString();
}

#endregion

public class Tokenizer : ITokenizer<ShakespeareToken>
{
    public IEnumerable<ShakespeareToken> Tokenize(Play play)
    {
        foreach (var act in play.Acts)
        {
            yield return ActChangeToken.Instance;

            foreach (var scene in act.Scenes)
            {
                yield return SceneChangeToken.Instance;

                if (scene.Dialog is null)
                    continue;

                foreach (var token in Tokenize(scene.Dialog, play.Characters))
                {
                    yield return token;
                }
            }
        }
    }

    public IEnumerable<ShakespeareToken> Tokenize(string corpus)
    {
        return Tokenize(corpus, null);
    }

    public IEnumerable<ShakespeareToken> Tokenize(string corpus, HashSet<string>? characterNames)
    {
        // Iterate word by word, emitting the correct tokens
        // "word" here refers to an actual word (space separated) or a punctuation mark

        StringBuilder buffer = new StringBuilder();
        int stageDirectionDepth = 0;

        string? flush_buffer()
        {
            if (buffer.Length == 0)
                return null;

            var str = buffer.ToString();
            buffer.Clear();
            return str;
        }

        ShakespeareToken wordOrSpeaker(string text)
        {
            if (stageDirectionDepth > 0)
                return new SpokenWord(text);

            if (text.Length > 1 && (text.All(char.IsUpper) || text.EndsWith(":")) && (characterNames is null || characterNames.Contains(text.TrimEnd(':'))))
            {
                return new SpeakerChangeToken(text.TrimEnd(':'));
            }
            else
            {
                return new SpokenWord(text);
            }
        }

        for (var i = 0; i < corpus.Length; i++)
        {
            char c = corpus[i];

            // Whitespace is a word separator
            if (char.IsWhiteSpace(c))
            {
                // Flush the buffer
                var str = flush_buffer(); if (str is not null) yield return wordOrSpeaker(str);
                continue;
            }

            // Character is punctuation
            bool isPunctuation = false;
            switch (c)
            {
                case '.':
                case '!':
                case '?':
                    // Terminal punctuation
                    var str = flush_buffer(); if (str is not null) yield return wordOrSpeaker(str);
                    yield return new TerminalPunctuation(c);
                    isPunctuation = true;
                    break;
                case ',':
                case ';':
                    // Pause punctuation
                    str = flush_buffer(); if (str is not null) yield return wordOrSpeaker(str);
                    yield return new PausePunctuation(c);
                    isPunctuation = true;
                    break;
            }
            if (isPunctuation)
                continue;

            // Stage directions
            bool isStageDirectionChar = false;
            switch (c)
            {
                case '[':
                    // Begin stage direction
                    var str = flush_buffer(); if (str is not null) yield return wordOrSpeaker(str);
                    stageDirectionDepth++;
                    yield return StageDirectionStart.Instance;
                    isStageDirectionChar = true;
                    break;
                case ']':
                    // End stage direction
                    str = flush_buffer(); if (str is not null) yield return wordOrSpeaker(str);
                    stageDirectionDepth = Math.Max(0, stageDirectionDepth - 1);
                    yield return StageDirectionEnd.Instance;
                    isStageDirectionChar = true;
                    break;
            }
            if (isStageDirectionChar)
                continue;

            // Else character is some kind of word or speaker change
            buffer.Append(c);
        }
    
        var strf = flush_buffer(); if (strf is not null) yield return wordOrSpeaker(strf);
    }
}