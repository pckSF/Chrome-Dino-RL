// Javascript code to extract the canvas from the div class="runner-canvas"
// Image width = 600px and height = 150px
var canvas = document.getElementsByClassName("runner-canvas")[0];
var base64_canvas = canvas.toDataURL("image/png");
return base64_canvas.replace("data:image/png;base64,", '');