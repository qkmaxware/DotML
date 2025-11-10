using System;
using System.Diagnostics.CodeAnalysis;

namespace Qkmaxware.Parsing {

public class Result<T> {

    [MemberNotNullWhen(true, nameof(Value))]
    [MemberNotNullWhen(false, nameof(Error))]
    public bool HasValue {get; private set;} = false;
    public T? Value {get; private set;}
    public Exception? Error {get; private set;} 
    public IInput Remainder {get; private set;}

    public Result(T value, Exception error, IInput remainder) {
        this.HasValue = value != null;
        this.Value = value;
        this.Error = error;
        this.Remainder = remainder;
    }

    public Result(T value, IInput remainder) {
        this.HasValue = value != null;
        this.Value = value;
        this.Error = null;
        this.Remainder = remainder;
    }

    public Result(Exception error, IInput remainder) {
        this.HasValue = false;
        this.Value = default(T);
        this.Error = error;
        this.Remainder = remainder;
    }
    
    public F Match<F>(Func<T?, F> success, Func<F> error)
    {
        if (this.HasValue) {
            return success(this.Value);
        } else {
            return error();
        }
    }

    public bool TryGetValue([NotNullWhen(true)]out T? value) {
        if (HasValue && this.Value != null) {
            value = this.Value;
            return true;
        } else {
            value = default(T);
            return false;
        }
    }

    public bool TryGetException([NotNullWhen(true)]out Exception? e) {
        if (HasValue) {
            e = default(Exception);
            return false;
        } else {
            e = this.Error ?? new Exception("Unknown parsing error");
            return true;
        }
    }

}

}