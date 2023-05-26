import argparse
from diffusers import DiffusionPipeline

parser = argparse.ArgumentParser(description='Generate an image using Stable Diffusion v1.5')
parser.add_argument('-i', action='store_true', help='Accept console/cli text input')

args = parser.parse_args()

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to("cuda")

print("What would you like to see?")
# img = pipeline("An image of a two chipmunks having dinner in Picasso style").images[0]
# img = pipeline("A squirrel holding a nut in the style of Joan Miro as if it's saying \"Thanks\"").images[0]
# img = pipeline("Two shrimp holding hands overlooking a sunset by the sea in the style of Monet").images[0]

# Accept input from command line
if args.i:
  input_text = input()
else:
  input_text = "A mystery"

img = pipeline(input_text).images[0]

# TODO: Use algorithm here to guess file name, and eventually ensure uniqueness with timestamp
input_filename = input('With what filename should I save it with?')
print('saving...')
img.save(input_filename)