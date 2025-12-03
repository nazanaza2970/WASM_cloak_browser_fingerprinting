from html_generator_regex import html_generator_regex

print(html_generator_regex("""// Get a reference to the canvas element
const canvas = document.getElementById('myCanvas');

// Get the 2D rendering context
const ctx = canvas.getContext('2d');

// Now you can use the 'ctx' object to draw on the canvas

// Example: Draw a red rectangle
ctx.fillStyle = 'red'; // Set the fill color to red
ctx.fillRect(50, 50, 100, 75); // Draw a rectangle at (50, 50) with width 100 and height 75

// Example: Draw a blue circle
ctx.beginPath(); // Start a new path
ctx.arc(250, 150, 40, 0, Math.PI * 2); // Create an arc (circle)
ctx.fillStyle = 'blue'; // Set the fill color to blue
ctx.fill(); // Fill the circle

// Example: Draw some text
ctx.font = '24px Arial'; // Set font style
ctx.fillStyle = 'black'; // Set text color
ctx.fillText('Hello Canvas!', 100, 250);"""))