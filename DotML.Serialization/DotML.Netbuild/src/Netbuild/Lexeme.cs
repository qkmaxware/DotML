using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

public abstract class Lexeme {
    public abstract Token? GetNext(int start_index, string str);
}

public class RegexLexeme : Lexeme {
    public Regex Pattern {get; private set;}
    public string? GroupName {get; protected set;}

    public RegexLexeme(string pattern, RegexOptions options) : this(new Regex(pattern, options)) {}
    public RegexLexeme(string pattern) : this(new Regex(pattern)) {}
    public RegexLexeme(Regex pattern) {
        this.Pattern = pattern;
    }

    public override Token? GetNext(int start_index, string str) {
        var match = NextMatch(start_index, str);
        if (match is null)
            return null;

        return new Token<string>(
            type: this, 
            position: start_index, 
            length: match.Length,
            value: (string.IsNullOrEmpty(GroupName) ? match.Value : match.Groups[GroupName].Value)
        );
    }

    public bool IsNextToken(int start_index, string str) {
        return this.Pattern.IsMatch(str, start_index);
    }

    public Match? NextMatch(int start_index, string str) {
        var match = this.Pattern.Match(str, start_index);
        if (match.Success)
            return match;
        return null;
    }
}