cs180 project 3: fun with flow models

what is here
parta.ipynb: part a. the course starter notebook with every exercise filled in. it uses the pretrained pixnerd model, so it needs a gpu.
partb.ipynb: part b. the unet, the one-step denoiser, and the time- and class-conditioned flow matching models on mnist.
parta_output.ipynb and partb_output.ipynb: the same notebooks after running on a colab t4, with every output saved. parta_output.ipynb is about 50 mb because it holds every image.
build_page.py: copies the figures the web page uses into media and writes index.html.
media: images for the web page.

how to run it in the colab web page
open a notebook in colab (file, upload notebook), choose a gpu (runtime, change runtime type, t4) and run all.
use a fresh runtime for each part. on a free t4, part a took about 1 hour 40 minutes and part b about 40 minutes.
each notebook ends by writing parta_results.zip or partb_results.zip into /content/cs180-hw3.

how i ran it
i used the colab cli from the terminal instead of the web page. the first command asks you to sign in to google.

    .venv/bin/pip install google-colab-cli
    .venv/bin/colab --auth oauth2 new -s partb --gpu T4
    .venv/bin/colab --auth oauth2 exec -s partb -f partb.ipynb --timeout 3600
    .venv/bin/colab --auth oauth2 download -s partb /content/cs180-hw3/partb_results.zip results/partb.zip
    .venv/bin/colab --auth oauth2 stop -s partb

then the same with parta. the free tier gives one gpu at a time, so the two parts run one after the other.
for part a i ran the velocity display cell (error limit 0.3) and the second hybrid cell afterwards in the same session, and copied their outputs into parta_output.ipynb.
unzip the results into results/parta and results/partb, then build the page:

    .venv/bin/python build_page.py

inputs
part a downloads two of my project 0 photos (doe library and the fire hydrant) from this repo on github, and the starry night from wikimedia commons.
the two drawings (a sailboat and mountains) are drawn by code in the notebook.
