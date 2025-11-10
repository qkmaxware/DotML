using System;
using System.Linq;
using System.Collections.Generic;

namespace Qkmaxware.Parsing {

/// <summary>
/// Static parsers for comments
/// </summary>
public static class Comment {
    /// <summary>
    /// Parse a C-style line comment
    /// </summary>
    public static Parser<string> Line() {
        return Text.Is("//").Then(Character.IsNot('\n').ZeroOrMore()).Map(chars => string.Concat(chars)).Named("Line Comment");
    }

    /// <summary>
    /// Parse a C-style block comment
    /// </summary>
    public static Parser<string> Block() {
        var commentChar = Character.IsNot('*').Or(Character.Is('*').UnlessFollowedBy(Character.Is('/')));
        var commentText = commentChar.ZeroOrMore().Map(digits => string.Concat(digits));
        return Text.Is("/*")
            .Then(
                commentText
            ) .Before(Text.Is("*/"))
            .Named("Block Comment");
    }

    private static Parser<T> Recursive<T>(Func<Parser<T>> func) {
        return item => func()(item);
    }

    /// <summary>
    /// Parse a C-style block comment which can include nested blocks
    /// </summary>
    public static Parser<string> NestedBlock() {
        var commentChar = 
            Character.IsNotOneOf('*', '/')                              // All characters
            .Or(Character.Is('*').UnlessFollowedBy(Character.Is('/')))  // Except */
            .Or(Character.Is('/').UnlessFollowedBy(Character.Is('*'))); // Except /* (this leads to a nested block)

        var commentText = commentChar.OneOrMore().Map(digits => string.Concat(digits));

        return Text.Is("/*")
            .Then(
                //commentText
                (commentText.Or(Recursive(NestedBlock))).ZeroOrMore().Map(lines => string.Concat(lines))
            ) .Before(Text.Is("*/"))
            .Named("Block Comment");
    }
}

}