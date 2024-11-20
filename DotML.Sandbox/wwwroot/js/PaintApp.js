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

    exports.init = function(canvas, width, height, import_input) {
        const context = canvas.getContext('2d');
        context.strokeStyle = "red";

        const MAIN_MOUSE_BUTTON = 0;

        shouldDraw = false;

        if (import_input) {
            function setup_import(canvas, input, object_fit) {
                function setup_contain(canvas, input) {
                    var load_image = function(e) {
                        var URL = window.URL;
                        if (!e.target.files || e.target.files.length < 1)
                            return;
                        var url = URL.createObjectURL(e.target.files[0]);
                        var image = new Image();
                        image.src = url;
                        image.onload = () => {
                            var ctx = canvas.getContext("2d");
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            if (image.width > image.height) {
                                var y_scale = canvas.width/image.width;
                                var y_offset = (canvas.height - (image.height * y_scale)) / 2.0;
                                ctx.drawImage(image, 0, y_offset, canvas.width, image.height * y_scale);
                            } else {
                                var x_scale = canvas.height/image.height;
                                var x_offset = (canvas.width - (image.width * x_scale)) / 2.0;
                                ctx.drawImage(image, x_offset, 0, image.width * x_scale, canvas.height);
                            }
                            canvas.dispatchEvent(new Event('mouseup', { bubbles: true, cancelable: true, view: window })); // Manually trigger mouse up as if the loaded image was "drawn"
                        }
                    }
                    input.addEventListener('change', load_image);
                }
                function setup_cover(canvas, input) {
                    var load_image = function(e) {
                        var URL = window.URL;
                        if (!e.target.files || e.target.files.length < 1)
                            return;
                        var url = URL.createObjectURL(e.target.files[0]);
                        var image = new Image();
                        image.src = url;
                        image.onload = () => {
                            var ctx = canvas.getContext("2d");
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            if (image.width > image.height) {
                                var x_scale = canvas.height/image.height;
                                var x_offset = (canvas.width - (image.width * x_scale)) / 2.0;
                                ctx.drawImage(image, x_offset, 0, image.width * x_scale, canvas.height);
                            } else {
                                var y_scale = canvas.width/image.width;
                                var y_offset = (canvas.height - (image.height * y_scale)) / 2.0;
                                ctx.drawImage(image, 0, y_offset, canvas.width, image.height * y_scale);
                            }
                            canvas.dispatchEvent(new Event('mouseup', { bubbles: true, cancelable: true, view: window })); // Manually trigger mouse up as if the loaded image was "drawn"
                        }
                    }
                    input.addEventListener('change', load_image);
                }
                
                switch (object_fit) {
                    case "contain":
                        setup_contain(canvas, input);
                        break;
                    case "cover":
                        setup_cover(canvas, input);
                        break;
                }
            }
            setup_import(canvas, import_input, 'cover');
        }

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
