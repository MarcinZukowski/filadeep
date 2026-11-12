FilaDeep

A tool (+website) to do some custom modifications to 3mf files, focused
on color printing.

3 identified functionalities so far, assuming you start with a
color-painted 3mf:

1.Create a separate-material internal core for the object. 
  Keep the external layers as are, keeping all the colors etc.
  Make the width of the outside configurable.
  Goal: Use single-color (white) strong material, and then a layer of colors 
  around it. Example: ASA inside + PETG outside, or PLA tought inside + PETG
  outside

2.Optimize the depths of the color parts for better visuals.
  Today, there are two related problems: 

2.1.When the model uses light or translucent colors, 
  but you add a dark element, like eyes, Bambu will often slice the eyes 
  very deep, and so they will seep through the light materials. 
  See "Deep plastic" in img/

2.2. When the model is dark, but you add light accents, 
  Bambu prints just 1 layer, which is not enough, see
  "Shallow plastic" im img/

So, the idea would be to introduce separate 3d models specifically in the
color of the print, and based on the color, make them shallow (for darks)
or deeper (for lights)

3.Add a layer around the print. Kind of the opposite of 1.
  Add a thin (e.g. 1mm) layer of additional material around the colored
  model (here model colors don't matter). This can be used to print a layer
  of PVB around the model, and then smooth it with alcohol, for example.

All problems seem to require similar infrastructure.

For now a Python tool is enough, but a web interface would be great.
