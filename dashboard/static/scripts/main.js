// function onClick(element) {
//   document.getElementById("img01").src = element.src;
//   document.getElementById("modal01").style.display = "block";
// }



var canvas = document.getElementById("DemoCanvas");
console.log(canvases[i])
if (canvas.getContext) {
var ctx = canvas.getContext('2d');
ctx.lineJoin = "round";
ctx.lineWidth = "25";
ctx.strokeStyle = "#EEF1FF";
ctx.strokeRect(20, 50, 250, 20);
ctx.strokeStyle = "#73A1FF";
ctx.strokeRect(150, 50, 50, 20);
ctx.lineJoin = "bevel";
ctx.lineWidth = "15";
ctx.strokeStyle = "#0021B6";
ctx.strokeRect(175, 46, 2, 28);
}


        
