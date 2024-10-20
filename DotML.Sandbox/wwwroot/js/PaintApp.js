if (!(window.BlazorComponents)) {
    window.BlazorComponents = {};
}

BlazorComponents.PaintApp = (function(){
    let exports = {};

    exports.clear = function(canvas) {
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height); 
    }

    exports.get_pixels = function(canvas, width, height) {
        const context = canvas.getContext('2d');
        var imaged = context.getImageData(0, 0, width, height).data;
        var pixels=[];
        for (var i = 0, n = imaged.length; i < n; i += 4) {
            pixels.push({
                R: imaged[i],
                G: imaged[i + 1],
                B: imaged[i + 2],
                // Alpha would be i+3
            });
        }
        return pixels;
    }

    exports.set_color = function(canvas, colour) {
        const context = canvas.getContext('2d');
        context.strokeStyle = colour;
    }

    exports.init = function(canvas, width, height) {
        const context = canvas.getContext('2d');
        context.strokeStyle = "red";

        const MAIN_MOUSE_BUTTON = 0;

        shouldDraw = false;

        function mousedown(event) {
            if (event.button === MAIN_MOUSE_BUTTON) {
                shouldDraw = true;
                context.beginPath();
                
                let elementRect = event.target.getBoundingClientRect();
                
                var clientX = event.clientX - elementRect.left;
                var clientY = event.clientY - elementRect.top;

                var canvasX = (clientX / elementRect.width) * width;
                var canvasY = (clientY / elementRect.height) * height;

                context.moveTo(canvasX, canvasY);
            }
        }

        function mouseup(event) {
            if (event.button === MAIN_MOUSE_BUTTON) {
                shouldDraw = false;
            }
        }

        function mousemove(event) {
            if (shouldDraw) {
                let elementRect = event.target.getBoundingClientRect();
                
                var clientX = event.clientX - elementRect.left;
                var clientY = event.clientY - elementRect.top;

                var canvasX = (clientX / elementRect.width) * width;
                var canvasY = (clientY / elementRect.height) * height;

                context.lineTo(canvasX, canvasY);
                context.stroke();
            }
        }

        canvas.addEventListener('mousedown', mousedown);
        canvas.addEventListener('mouseup', mouseup);
        canvas.addEventListener('mousemove', mousemove);
    }

    return exports;
})();
