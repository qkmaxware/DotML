using System;
using System.Collections.Generic;

namespace Qkmaxware.Parsing {

/// <summary>
/// Various parser combinator extention methods
/// </summary>
public static class Combinators {

    // ---------------------------------------------------------------------------------------
    // Data cleaning
    // ---------------------------------------------------------------------------------------

    /// <summary>
    /// Trim whitespace from the front of the input before running the parser
    /// </summary>
    public static Parser<T> TrimLeft<T>(this Parser<T> parser) {
        return (Character.Whitespace().ZeroOrMore()).Then(parser);
    }

    /// <summary>
    /// Trim whitespace from the end of the input before running the parser
    /// </summary>
    public static Parser<T> TrimRight<T>(this Parser<T> parser) {
        return parser.Before(Character.Whitespace().ZeroOrMore());
    }

    /// <summary>
    /// Trim whitespace from both ends of the input before running the parser
    /// </summary>
    public static Parser<T> Trim<T>(this Parser<T> parser) {
        return ((Character.Whitespace().ZeroOrMore()).Then(parser)).Before(Character.Whitespace().ZeroOrMore());
    }

    // ---------------------------------------------------------------------------------------
    // Combinations
    // ---------------------------------------------------------------------------------------

    /// <summary>
    /// Sequence two parsers, keep the results of the second
    /// </summary>
    public static Parser<U> Then<T,U> (this Parser<T> first, Parser<U> second) {
        return input => {
            var rf = first(input);
            if (!rf.HasValue)
                return new Result<U>(rf.Error, input);
            return second(rf.Remainder);
        };
    }
    
    /// <summary>
    /// Sequence two parsers, keep the results of the second
    /// </summary>
    public static Parser<O> ThenMap<T,U,O> (this Parser<T> first, Parser<U> second, Func<T, U, O> mapping) {
        return input => {
            var rf = first(input);
            if (!rf.HasValue)
                return new Result<O>(rf.Error, input);
            var lf = second(rf.Remainder);
            if (!lf.HasValue)
                return new Result<O>(lf.Error, input);
            return new Result<O>(mapping(rf.Value, lf.Value), lf.Remainder);
        };
    }

    /// <summary>
    /// Sequence two parsers, keep the results of the second
    /// </summary>
    public static Parser<U> Then<T, U>(this Parser<T> first, Func<T, Parser<U>> second)
    {
        return input =>
        {
            var rf = first(input);
            if (!rf.HasValue)
                return new Result<U>(rf.Error, input);
            return second(rf.Value)(rf.Remainder);
        };
    }

    /// <summary>
    /// Sequence two parsers, keep the results of the first
    /// </summary>
    public static Parser<T> Before<T,U> (this Parser<T> first, Parser<U> second) {
        return input => {
            var rf = first(input);
            if (!rf.HasValue)
                return new Result<T>(rf.Error, input);
            var rf2 = second(rf.Remainder);
            if (!rf2.HasValue)
                return new Result<T>(rf2.Error, rf.Remainder);
            return new Result<T>(rf.Value, rf2.Remainder);
        };
    }

    /// <summary>
    /// Sequence parsers, keeping the middle result
    /// </summary>
    public static Parser<T> Between<A, T, B> (this Parser<T> parser, Parser<A> lhs, Parser<B> rhs) {
        return input => {
            var r1 = lhs(input);
            if (!r1.HasValue)
                return new Result<T>(r1.Error, input);
            
            var result = parser(r1.Remainder);
            if (!result.HasValue)
                return new Result<T>(result.Error, r1.Remainder);

            var r2 = rhs(result.Remainder);
            if (!r2.HasValue)
                return new Result<T>(r2.Error, result.Remainder);

            return result;
        };
    }

    public static Parser<Maybe<T>> Optional<T>(this Parser<T> parser) {
        return input =>
            {
                var result = parser(input);
                if (!result.HasValue)
                    return new Result<Maybe<T>>(new Maybe<T>.None(), input);
                return new Result<Maybe<T>>(new Maybe<T>.Some(result.Value), result.Remainder);
            };
    }

    public static Parser<U> Map<T,U>(this Parser<T> parser, Func<T, U> mapping) {
        return input => {
            var tresult = parser(input);
            if (!tresult.HasValue)
                return new Result<U>(tresult.Error, input);
            return new Result<U>(mapping(tresult.Value), tresult.Remainder);
        };
    }

    public static Parser<T> Unless<T>(this Parser<T> parser, Func<T, bool> condition) {
        return input => {
            // Read in result
            var tresult = parser(input);
            if (!tresult.HasValue)
                return new Result<T>(tresult.Error, input);
            // Test condition
            if (condition(tresult.Value)) {
                // Condition is true, BAD
                return new Result<T>(new ArgumentException("Parsed value failed to meet the condition"), input);
            } else {
                // Condition is false, GOOD
                return tresult;
            }
        };
    }

    public static Parser<T> UnlessFollowedBy<T,U>(this Parser<T> parser, Parser<U> condition) {
        return input => {
            // Read in result
            var tresult = parser(input);
            if (!tresult.HasValue)
                return new Result<T>(tresult.Error, input);
            // Test condition
            var follow = condition(tresult.Remainder);
            if (!follow.HasValue)
                return tresult;
            else 
                return new Result<T>(new ArgumentException("Parsed value followed by invalid input"), input);
        };
    }

    public static Parser<T> Or<T> (this Parser<T> first, Parser<T> second) {
        return input => {
            var result = first(input);
            if (result.HasValue)
                return result;
            return second(input);
        };
    }

    public static Parser<List<T>> ZeroOrMore<T>(this Parser<T> item) {
        return input => {
            var many = new List<T>();
            var next = item(input);
            while (next.HasValue) {
                many.Add(next.Value);
                next = item(next.Remainder);
            }
            return new Result<List<T>>(many, next.Remainder);
        };
    }

    public static Parser<List<T>> ZeroOrMoreSeparatedBy<T,U>(this Parser<T> item, Parser<U> separator) {
        return input => {
            var many = new List<T>();

            // First item
            var next = item(input);
            if (!next.HasValue) 
                return new Result<List<T>>(many, input);

            // Remainder
            var sep = separator(next.Remainder);
            while(sep.HasValue) {
                next = item(sep.Remainder);
                if (!next.HasValue)
                    return new Result<List<T>>(next.Error, sep.Remainder);
                many.Add(next.Value);

                sep = separator(next.Remainder);
            }
            return new Result<List<T>>(many, next.Remainder);
        };
    }

    public static Parser<List<T>> OneOrMore<T>(this Parser<T> item) {
        return input => {
            var many = new List<T>();

            // First item
            var next = item(input);
            if (!next.HasValue) 
                return new Result<List<T>>(next.Error, input);
            many.Add(next.Value);
            
            // Remainder
            next = item(next.Remainder);
            while (next.HasValue) {
                many.Add(next.Value);
                next = item(next.Remainder);
            }
            return new Result<List<T>>(many, next.Remainder);
        };
    }

    public static Parser<List<T>> OneOrMoreSeparatedBy<T,U>(this Parser<T> item, Parser<U> separator) {
        return input => {
            var many = new List<T>();

            // First item
            var next = item(input);
            if (!next.HasValue) 
                return new Result<List<T>>(next.Error, input);
            many.Add(next.Value);

            // Remainder
            var sep = separator(next.Remainder);
            while(sep.HasValue) {
                next = item(sep.Remainder);
                if (!next.HasValue)
                    return new Result<List<T>>(next.Error, sep.Remainder);
                many.Add(next.Value);
                
                sep = separator(next.Remainder);
            }
            return new Result<List<T>>(many, next.Remainder);
        };
    }

    public static Parser<List<T>> Repeat<T>(this Parser<T> parser, int n) {
        return input => {
            var many = new List<T>();
            Result<T>? last = null;
            for (int i = 0; i < n; i++) {
                var next = parser(input);
                if (!next.HasValue) 
                    return new Result<List<T>>(next.Error, last?.Remainder ?? input);
                last = next;
            }
            return new Result<List<T>>(many, last?.Remainder ?? input);
        };
    }

    public static Parser<T> Named<T> (this Parser<T> parser, string name) {
        return input => {
            var next = parser(input);
            if (!next.HasValue) 
                return new Result<T>(new Exception($"Missing or invalid '{name}'", next.Error), input);
            return next;
        };
    }

}

}