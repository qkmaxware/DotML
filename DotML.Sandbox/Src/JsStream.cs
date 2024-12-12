using Microsoft.JSInterop;

namespace DotML.Sandbox.Data;

/// <summary>
/// Class to stream data from JavaScript
/// </summary>
public class JsStream
{
    private List<byte> _data;
    public delegate void StreamFlushHandler(System.Collections.ObjectModel.ReadOnlyCollection<byte> bytes);
    public event StreamFlushHandler OnStreamFlush = delegate { };
    public JsStream() {
        this._data = new List<byte>();
    }

    public JsStream(int capacity) {
        this._data = new List<byte>(capacity);
    }

    [JSInvokable]
    public void Write(byte[] chunk) {
        this._data.AddRange(chunk);
    }

    [JSInvokable]
    public void Flush() {
        OnStreamFlush?.Invoke(this._data.AsReadOnly());
    }

    public DotNetObjectReference<JsStream> GetJsWriter() {
        return DotNetObjectReference.Create(this);
    }

}