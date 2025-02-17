# PyHHB-HS1
A preliminary repository for distributing a novel and robust human heat balance model

To improve upon existing methods of quantifying heat stress, Vanos et al (2023) created a novel model of human heat balance (https://doi.org/10.1038/s41467-023-43121-5) that's great for use with climate model output and can support multiple physiologies. This model is internally referred to as PyHHB (the Python Human Heat Balance model), but the model could theoretically be implemented in other languages just as well. In any case, PyHHB is great, so it is high time it be distributed to the public. This practice repository is a place for me to practice using GitHub to share my improvements to this model. There's a different GitHub where PyHHB folks are working on merging together all our various little QOL tweaks, but we're not quite ready to share that yet.

A *very* brief overview of the content of this GitHub . . . feel free to reach out if you have more questions!

`HHB.py`: All the various functions used in PyHHB are defined here. This is like the bones of PyHHB, but if all the bones were kind of haphazardly arranged in a pile.
`a_single_function.ipynb`: This is like if you take all those bones and arrange them into a functioning skeleton. Here, I've taken all the various PyHHB functions, and arranged them into a single, easy-to-use function that computes livability and survivability at the same time.
`00_create_lookup_table.py`: Unfortunately, PyHHB is not yet vectorized. We're working on that, but in the meantime, it can be really slow to use with giant datasets (e.g. climate model output). To speed things up, you can simply create a 3D "look-up table" in the temperature-humidity-pressure space, such that you can simply look up what the answer will be at a given point in that space as opposed to running the entire model every time. Note that you'll need to create different look-up tables for different physiologies!
`05_run_look_up_table_pyhhb.py`: Here's a sample script of how I'm putting the look-up table into action with my CMIP output.
