using System;

namespace Qkmaxware.Parsing {

public class Result<T> {
    
    public bool HasValue {get; private set;} = false;
    public T Value {get; private set;}
    public Exception Error {get; private set;} 
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

    public bool TryGetValue(out T value) {
        if (HasValue) {
            value = this.Value;
            return true;
        } else {
            value = default(T);
            return false;
        }
    }

    public bool TryGetException(out Exception e) {
        if (HasValue) {
            e = default(Exception);
            return false;
        } else {
            e = this.Error;
            return true;
        }
    }

}

}