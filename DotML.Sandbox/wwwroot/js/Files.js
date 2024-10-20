if (!(window.BlazorComponents)) {
    window.BlazorComponents = {};
}

BlazorComponents.Files = (function(){
    let exports = {};

    exports.download = function(filename, content_type, data) {
        // Create the URL
        const file = new File([data], filename, { type: content_type });
        const exportUrl = URL.createObjectURL(file);

        // Create the <a> element and click on it
        const a = document.createElement("a");
        document.body.appendChild(a);
        a.href = exportUrl;
        a.download = filename;
        a.target = "_self";
        a.click();

        // We don't need to keep the url, let's release the memory
        URL.revokeObjectURL(exportUrl);
        a.remove();
    }

    return exports;
})();
