using System;
using System.Runtime.InteropServices;

namespace Qkmaxware.Parsing {

/// <summary>
/// Static parsers for strings
/// </summary>
public static class Text
{
    /// <summary>
    /// Parse the given string
    /// </summary>
    public static Parser<string> Is(string str)
    {
        return input =>
        {
            var currentStep = input;
            foreach (var c in str)
            {
                var next = currentStep.NextChar();
                char parsed;
                if (next.TryGetValue(out parsed))
                {
                    if (parsed == c)
                    {
                        // Char is RIGHT
                        currentStep = next.Remainder;
                        continue;
                    }
                    else
                    {
                        // Char is wrong
                        return new Result<string>(new ArgumentException($"Expecting {c} in string {str} but found {parsed}"), input);
                    }
                }
                else
                {
                    // Char not found
                    return new Result<string>(new ArgumentException("End of character stream"), input);
                }
            }
            return new Result<string>(str, currentStep);
        };
    }

    /// <summary>
    /// Create a string from a list of characters
    /// </summary>
    /// <param name="characters">parser that creates a list of characters</param>
    public static Parser<string> FromCharacters(Parser<List<char>> characters)
    {
        return input =>
        {
            var next = characters(input);
            if (!next.HasValue)
                return new Result<string>(next.Error, input);

            return new Result<string>(new string(CollectionsMarshal.AsSpan(next.Value)), next.Remainder);
        };
    }

    private static Parser<string> Quoted(char quoteChar)
    {
        var quote = Character.Is(quoteChar);
        var backslash = Character.Is('\\');

        var escapedChar = backslash.Then(Character.Any()).Map(escaped => escaped switch
        {
            '"' => '"',
            '\\' => '\\',
            '/' => '/',
            'b' => '\b',
            'f' => '\f',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            _ => '?' // Unrecongized escape sequence. Could fail if I really want to, decide later
        });
        var normalChar = Character.IsNot('\\');

        var contentChar = escapedChar.Or(normalChar);
        var content = Text.FromCharacters(contentChar.ZeroOrMore()); // string

        return content.Between(quote, quote).Named("Quoted string");
    }

    public static Parser<string> DoubleQuoted() => Quoted('"');
    public static Parser<string> SingleQuoted() => Quoted('"');
}

}